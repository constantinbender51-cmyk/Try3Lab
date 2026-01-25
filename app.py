import os
import ccxt
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt
import io
import base64
import http.server
import socketserver
import threading
import time
import json
from datetime import datetime, timedelta

# --- Configuration ---
TIMEFRAME = os.environ.get('TIMEFRAME', '1h')
SYMBOL = os.environ.get('SYMBOL', 'BTC/USDT')
START = os.environ.get('START', '2024-01-01 00:00:00')
END = os.environ.get('END', '2024-06-01 00:00:00')

# Parameters
A = float(os.environ.get('A', 0.0))          # Unused (rounding removed)
B = float(os.environ.get('B', 0.7))          # Split % (70% training, 30% testing)
C = float(os.environ.get('C', 0.1))          # Top % most frequent (densest) sequences to keep
D = int(os.environ.get('D', 4))              # Sequence length (candles)
E = float(os.environ.get('E', 0.002))        # Similarity threshold (0.1% absolute diff)

# Global State
results_html = "<h1>Initializing...</h1>"
live_outcomes = []
model_sequences = None
data_global = None

# API Data Containers
backtest_data = {}
recent_data = {}

# --- Functions ---

def fetch(timeframe, symbol, start_str, end_str):
    """Fetches OHLCV data from Binance."""
    print(f"Fetching {symbol} {timeframe} from {start_str} to {end_str}...")
    exchange = ccxt.binance()
    start_ts = exchange.parse8601(start_str)
    end_ts = exchange.parse8601(end_str)
    
    ohlc = []
    current_ts = start_ts
    
    while current_ts < end_ts:
        try:
            candles = exchange.fetch_ohlcv(symbol, timeframe, since=current_ts, limit=1000)
            if not candles:
                break
            
            # Filter out candles beyond end_ts
            candles = [c for c in candles if c[0] < end_ts]
            if not candles:
                break

            ohlc += candles
            current_ts = candles[-1][0] + 1
            time.sleep(0.1) 
        except Exception as e:
            print(f"Error fetching: {e}")
            break
            
    df = pd.DataFrame(ohlc, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def deriveround(df, a=None):
    """Applies returns calculation."""
    df = df.copy()
    cols = ['open', 'high', 'low', 'close']
    for col in cols:
        df[f'{col}_ret'] = df[col].pct_change()
    
    df.dropna(inplace=True)
    return df

def split(df, b):
    split_idx = int(len(df) * b)
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()

def gettop(df_split1, c, d):
    """Finds dense patterns in Split 1."""
    print("Training model (finding dense patterns)...")
    
    data_cols = ['open_ret', 'high_ret', 'low_ret', 'close_ret']
    data_values = df_split1[data_cols].values
    
    # Shape: (N_windows, D, 4)
    windows = np.lib.stride_tricks.sliding_window_view(data_values, window_shape=d, axis=0)
    
    N = windows.shape[0]
    flat_windows = windows.reshape(N, -1)
    
    densities = np.zeros(N, dtype=int)
    chunk_size = 1000
    
    for i in range(0, N, chunk_size):
        end = min(i + chunk_size, N)
        batch = flat_windows[i:end]
        
        # Check distance against ALL windows (sampled or blocked if too large)
        compare_set = flat_windows if N < 10000 else flat_windows[::5]
        
        for j in range(len(batch)):
            diff = np.abs(compare_set - batch[j])
            matches = np.all(diff < E, axis=1)
            densities[i+j] = np.sum(matches)
            
    top_n = int(N * c)
    if top_n == 0: top_n = 1
    
    top_indices = np.argsort(densities)[-top_n:]
    return windows[top_indices]

def completesimilarbeginnings(df_target, model_patterns, e, d):
    """Predicts on df_target using the top patterns."""
    print("Running predictions...")
    data_cols = ['open_ret', 'high_ret', 'low_ret', 'close_ret']
    ohlc_cols = ['open', 'high', 'low', 'close']
    
    target_values = df_target[data_cols].values
    timestamps = df_target['timestamp'].values
    ohlc_values = df_target[ohlc_cols].values
    
    # Extract raw close prices for entry/exit logging
    close_prices = df_target['close'].values
    
    target_windows = np.lib.stride_tricks.sliding_window_view(target_values, window_shape=d, axis=0)
    target_ts = np.lib.stride_tricks.sliding_window_view(timestamps, window_shape=d, axis=0)
    target_ohlc = np.lib.stride_tricks.sliding_window_view(ohlc_values, window_shape=d, axis=0)
    target_prices = np.lib.stride_tricks.sliding_window_view(close_prices, window_shape=d, axis=0)
    
    predictions = []
    
    model_context = model_patterns[:, :d-1, :] # (K, D-1, 4)
    model_outcome = model_patterns[:, -1, :]   # (K, 4)
    model_context_flat = model_context.reshape(model_context.shape[0], -1)
    
    for i in range(len(target_windows)):
        current_window = target_windows[i] # Shape (D, 4)
        current_context = current_window[:d-1, :]
        current_context_flat = current_context.reshape(-1)
        
        diff = np.abs(model_context_flat - current_context_flat)
        matches_idx = np.where(np.all(diff < e, axis=1))[0]
        
        if len(matches_idx) > 0:
            matched_outcomes = model_outcome[matches_idx]
            avg_return = np.mean(matched_outcomes[:, 3]) # Column 3 is close_ret
            
            predicted_dir = 1 if avg_return > 0 else -1
            if avg_return == 0: predicted_dir = 0
            
            # Outcome
            # The window is [t-3, t-2, t-1, t]. 
            # We predict at t-1 (entry).
            # Outcome is at t.
            actual_ret = current_window[-1, 3] 
            actual_dir = 1 if actual_ret > 0 else -1
            if actual_ret == 0: actual_dir = 0
            
            # --- FIX: Align Entry Time to the Start of the Outcome Candle ---
            # Index -1 corresponds to the outcome candle (the one we are predicting)
            ts = pd.to_datetime(target_ts[i, -1])
            
            # Prices
            # Entry price is the Close of the last context candle (Index -2)
            # or equivalently the Open of the outcome candle (Index -1)
            # We use Close of previous (-2) to simulate instant fill on candle close.
            entry_price = target_prices[i, -2]
            exit_price = target_prices[i, -1]
            
            # Input Sequence (timestamps and OHLC)
            # Indices 0 to D-2 (inclusive) are the Context.
            input_timestamps = pd.to_datetime(target_ts[i, :d-1]).strftime('%Y-%m-%d %H:%M:%S').tolist()
            input_candles = target_ohlc[i, :d-1].tolist()

            predictions.append({
                'timestamp': ts,
                'predicted_dir': predicted_dir,
                'actual_ret': actual_ret,
                'actual_dir': actual_dir,
                'is_correct': (predicted_dir == actual_dir) and (predicted_dir != 0),
                'entry_price': entry_price,
                'exit_price': exit_price,
                'input_timestamps': input_timestamps,
                'input_candles': input_candles
            })
            
    return pd.DataFrame(predictions)

def printaccuracy(predictions_df):
    """
    Generates HTML report and structured data dictionary.
    """
    if predictions_df.empty:
        return "<h3>No predictions made (adjust E or C)</h3>", {"error": "No predictions"}

    active = predictions_df[predictions_df['predicted_dir'] != 0].copy()
    if active.empty:
        return "<h3>No non-flat predictions</h3>", {"error": "No active predictions"}

    total = len(active)
    correct = active['is_correct'].sum()
    accuracy = (correct / total) * 100
    
    active['pnl'] = active['predicted_dir'] * active['actual_ret']
    active['cum_pnl'] = active['pnl'].cumsum()
    
    # --- Plot Generation ---
    plt.figure(figsize=(10, 5))
    plt.plot(active['timestamp'], active['cum_pnl'], label='Cumulative PnL (Strategy)')
    plt.title(f'Strategy Performance (Acc: {accuracy:.2f}%)')
    plt.grid(True)
    plt.legend()
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    # --- HTML Generation ---
    table_html = """
    <table border="1">
    <tr><th>Entry Time</th><th>Pred</th><th>Entry Price</th><th>Exit Price</th><th>Actual Ret</th><th>Outcome</th><th>PnL</th><th>Input Context (Last 3 Candles)</th></tr>
    """
    for _, row in active.tail(50).iterrows():
        p_str = "UP" if row['predicted_dir'] > 0 else "DOWN"
        color = "green" if row['is_correct'] else "red"
        
        # Format input candles
        inputs_str = "<div style='font-size:0.8em'>"
        if isinstance(row['input_candles'], list) and isinstance(row['input_timestamps'], list):
            for t, c in zip(row['input_timestamps'], row['input_candles']):
                # Round candle values for display
                c_rounded = [round(x, 2) for x in c]
                inputs_str += f"{t}: {c_rounded}<br>"
        inputs_str += "</div>"
        
        table_html += f"<tr><td>{row['timestamp']}</td><td>{p_str}</td><td>{row['entry_price']:.2f}</td><td>{row['exit_price']:.2f}</td><td>{row['actual_ret']:.4f}</td><td style='color:{color}'>{row['is_correct']}</td><td>{row['pnl']:.4f}</td><td>{inputs_str}</td></tr>"
    table_html += "</table>"
    
    html_out = f"<h3>Accuracy: {accuracy:.2f}% ({correct}/{total})</h3><img src='data:image/png;base64,{plot_url}'/><br>{table_html}"

    # --- API Data Generation ---
    equity_curve = active[['timestamp', 'cum_pnl']].copy()
    equity_curve['timestamp'] = equity_curve['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    trade_cols = ['timestamp', 'predicted_dir', 'entry_price', 'exit_price', 'actual_ret', 'is_correct', 'pnl', 'input_timestamps', 'input_candles']
    trade_history = active[trade_cols].copy()
    trade_history['timestamp'] = trade_history['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    stats_data = {
        "accuracy_percent": round(accuracy, 2),
        "total_trades": int(total),
        "correct_trades": int(correct),
        "cumulative_pnl": float(active['cum_pnl'].iloc[-1]),
        "equity_curve": equity_curve.to_dict(orient='records'),
        "trade_history": trade_history.to_dict(orient='records'),
        "plot_base64": plot_url
    }
    
    return html_out, stats_data

def predict_on_recent(model_patterns, df_recent, e, d):
    preds = completesimilarbeginnings(df_recent, model_patterns, e, d)
    return preds

# --- Live Loop ---

def get_seconds_to_sleep(timeframe):
    now = datetime.utcnow()
    unit = timeframe[-1]
    val = int(timeframe[:-1])
    
    if unit == 'm': delta = timedelta(minutes=val)
    elif unit == 'h': delta = timedelta(hours=val)
    elif unit == 'd': delta = timedelta(days=val)
    else: delta = timedelta(hours=1)

    # Calculate next close alignment
    if unit == 'h':
        next_hour = now.replace(minute=0, second=0, microsecond=0) + delta
        while next_hour < now: next_hour += delta
        target = next_hour
    elif unit == 'm':
        next_min = now.replace(second=0, microsecond=0) + timedelta(minutes=val - (now.minute % val))
        if next_min <= now: next_min += timedelta(minutes=val)
        target = next_min
    else:
        target = now + delta

    seconds = (target - now).total_seconds() + 5 
    return max(0, seconds)

def live_loop():
    global live_outcomes
    while True:
        try:
            sec = get_seconds_to_sleep(TIMEFRAME)
            print(f"Live Loop: Sleeping {sec:.1f}s until next close...")
            time.sleep(sec)
            
            print("Live Loop: Fetching latest data...")
            exchange = ccxt.binance()
            limit = D * 5
            candles = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=limit)
            df_live = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df_live['timestamp'] = pd.to_datetime(df_live['timestamp'], unit='ms')
            
            df_derived = deriveround(df_live)
            
            # Step 1: Resolve previous prediction
            if len(live_outcomes) > 0 and 'outcome' not in live_outcomes[-1]:
                last_pred = live_outcomes[-1]
                
                # Check if we have the specific close required
                # Last pred was made at df_live.iloc[-2] (conceptually)
                # We need the close of the next candle
                
                if len(df_derived) >= 2:
                    # Current last completed candle is -1
                    # If this matches the prediction interval
                    
                    actual_close_ret = df_derived.iloc[-1]['close_ret'] # Most recent completed return
                    
                    # We need to verify if this is indeed the candle AFTER the entry
                    # In live loop, we wake up after a candle closes.
                    # df_live[-1] is the just-closed candle.
                    
                    exit_price = df_live.iloc[-1]['close']
                    entry_price = last_pred.get('entry_price', exit_price)
                    
                    # Simple PnL calc
                    raw_return = (exit_price - entry_price) / entry_price if entry_price > 0 else 0
                    actual_dir = 1 if raw_return > 0 else -1
                    if raw_return == 0: actual_dir = 0
                    
                    if last_pred['pred_dir'] == 0:
                        last_pred['outcome'] = "Flat"
                        last_pred['pnl'] = 0.0
                    else:
                        last_pred['outcome'] = (last_pred['pred_dir'] == actual_dir)
                        last_pred['pnl'] = last_pred['pred_dir'] * raw_return

                    last_pred['actual_ret'] = raw_return
                    last_pred['exit_price'] = exit_price
            
            # Step 2: Make NEW prediction
            # We use the LAST D candles as the window. 
            # D-1 candles are context. 
            # We predict the NEXT candle (which hasn't happened yet).
            # Wait, the model predicts the *last* component of a sequence of D.
            # So we supply D-1 inputs, and predict D.
            
            if len(df_derived) >= D:
                # We want to predict the candle starting NOW.
                # So we take the last D-1 completed candles as context.
                
                # df_derived includes returns for the just-closed candle at index -1.
                # So context is df_derived.iloc[-(D-1):]
                
                context_slice = df_derived.iloc[-(D-1):]
                recent_context = context_slice[['open_ret', 'high_ret', 'low_ret', 'close_ret']].values
                recent_context_flat = recent_context.reshape(-1)
                
                # Extract input sequence info (Raw OHLC)
                # These are the candles corresponding to the returns in context_slice
                # context_slice indices match df_live indices (if aligned).
                # deriveround drops index 0.
                
                input_rows = df_live.loc[context_slice.index]
                input_timestamps = input_rows['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist()
                input_candles = input_rows[['open', 'high', 'low', 'close']].values.tolist()
                
                model_context = model_sequences[:, :D-1, :]
                model_context_flat = model_context.reshape(model_context.shape[0], -1)
                
                diff = np.abs(model_context_flat - recent_context_flat)
                matches_idx = np.where(np.all(diff < E, axis=1))[0]
                
                # Entry price is the close of the last completed candle
                entry_price = df_live.iloc[-1]['close']
                
                if len(matches_idx) > 0:
                    model_outcomes = model_sequences[matches_idx, -1, :]
                    avg_ret = np.mean(model_outcomes[:, 3])
                    pred_dir = 1 if avg_ret > 0 else -1
                    if avg_ret == 0: pred_dir = 0
                    
                    live_outcomes.append({
                        'time': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
                        'pred_dir': pred_dir,
                        'matches': int(len(matches_idx)), # ensure int for JSON
                        'entry_price': entry_price,
                        'input_timestamps': input_timestamps,
                        'input_candles': input_candles
                    })
                else:
                    live_outcomes.append({
                        'time': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
                        'pred_dir': 0,
                        'matches': 0,
                        'note': 'No Match',
                        'entry_price': entry_price,
                        'input_timestamps': input_timestamps,
                        'input_candles': input_candles
                    })
            
            if len(live_outcomes) > 336:
                live_outcomes.pop(0)
                
        except Exception as e:
            print(f"Live loop error: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(60)

# --- Web Server & API ---

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        global results_html, live_outcomes, backtest_data, recent_data
        
        # --- API Routes ---
        if self.path == '/api/current':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            data = live_outcomes[-1] if live_outcomes else {"status": "waiting for data"}
            self.wfile.write(json.dumps(data, cls=CustomJSONEncoder).encode())
            return
            
        elif self.path == '/api/live':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(live_outcomes, cls=CustomJSONEncoder).encode())
            return
            
        elif self.path == '/api/backtest':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(backtest_data, cls=CustomJSONEncoder).encode())
            return
            
        elif self.path == '/api/recent':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(recent_data, cls=CustomJSONEncoder).encode())
            return

        # --- Default HTML Route ---
        live_html = "<h2>Live Outcomes (Last 2 weeks)</h2><table border='1'><tr><th>Time</th><th>Pred</th><th>Matches</th><th>Entry</th><th>Exit</th><th>Outcome</th><th>PnL</th><th>Input Context</th></tr>"
        for item in reversed(live_outcomes):
            outcome_str = item.get('outcome', 'Pending...')
            pnl_str = f"{item.get('pnl', 0):.4f}" if 'pnl' in item else "-"
            entry_s = f"{item.get('entry_price', 0):.2f}"
            exit_s = f"{item.get('exit_price', 0):.2f}" if 'exit_price' in item else "-"
            
            # Format input candles
            inputs_str = "<div style='font-size:0.8em'>"
            if 'input_candles' in item and 'input_timestamps' in item:
                for t, c in zip(item['input_timestamps'], item['input_candles']):
                    c_rounded = [round(x, 2) for x in c]
                    inputs_str += f"{t}: {c_rounded}<br>"
            inputs_str += "</div>"
            
            pred_s = "UP" if item['pred_dir'] == 1 else ("DOWN" if item['pred_dir'] == -1 else "FLAT")
            live_html += f"<tr><td>{item['time']}</td><td>{pred_s}</td><td>{item['matches']}</td><td>{entry_s}</td><td>{exit_s}</td><td>{outcome_str}</td><td>{pnl_str}</td><td>{inputs_str}</td></tr>"
        live_html += "</table>"
        
        full_page = f"""
        <html><head><title>Pattern Matcher</title>
        <meta http-equiv="refresh" content="30">
        </head><body>
        <h1>Market Pattern Matcher: {SYMBOL} {TIMEFRAME}</h1>
        <p>API Endpoints: <a href="/api/current">/api/current</a>, <a href="/api/live">/api/live</a>, <a href="/api/backtest">/api/backtest</a>, <a href="/api/recent">/api/recent</a></p>
        {results_html}
        <hr>
        {live_html}
        </body></html>
        """
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(full_page.encode())

def run_server():
    with socketserver.TCPServer(("", 8080), Handler) as httpd:
        print("Serving on port 8080")
        httpd.serve_forever()

# --- Main ---

def main():
    global model_sequences, results_html, data_global, backtest_data, recent_data
    
    df = fetch(TIMEFRAME, SYMBOL, START, END)
    df_derived = deriveround(df, A)
    data_global = df_derived
    
    split1, split2 = split(df_derived, B)
    print(f"Split 1 size: {len(split1)}, Split 2 size: {len(split2)}")
    
    model_sequences = gettop(split1, C, D)
    print(f"Model trained. {len(model_sequences)} patterns retained.")
    
    preds = completesimilarbeginnings(split2, model_sequences, E, D)
    
    recent_start = (datetime.utcnow() - timedelta(days=14)).isoformat()
    recent_end = datetime.utcnow().isoformat()
    df_recent = fetch(TIMEFRAME, SYMBOL, recent_start, recent_end)
    df_recent_derived = deriveround(df_recent)
    preds_recent = predict_on_recent(model_sequences, df_recent_derived, E, D)
    
    # Store both HTML and Data
    html_split2, backtest_data = printaccuracy(preds)
    html_recent, recent_data = printaccuracy(preds_recent)
    
    results_html = f"""
    <h2>Backtest (Split 2)</h2>
    {html_split2}
    <hr>
    <h2>Recent 14 Days Performance</h2>
    {html_recent}
    """
    
    t_live = threading.Thread(target=live_loop)
    t_live.daemon = True
    t_live.start()
    
    run_server()

if __name__ == "__main__":
    main()