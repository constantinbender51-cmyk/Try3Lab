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
    target_values = df_target[data_cols].values
    timestamps = df_target['timestamp'].values
    
    # Extract raw close prices for entry/exit logging
    close_prices = df_target['close'].values
    
    target_windows = np.lib.stride_tricks.sliding_window_view(target_values, window_shape=d, axis=0)
    target_ts = np.lib.stride_tricks.sliding_window_view(timestamps, window_shape=d, axis=0)
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
            actual_ret = current_window[-1, 3] 
            actual_dir = 1 if actual_ret > 0 else -1
            if actual_ret == 0: actual_dir = 0
            
            ts = pd.to_datetime(target_ts[i, -1])
            
            # Prices
            # Entry: Close of the candle BEFORE the outcome (index -2)
            # Exit: Close of the outcome candle (index -1)
            entry_price = target_prices[i, -2]
            exit_price = target_prices[i, -1]
            
            predictions.append({
                'timestamp': ts,
                'predicted_dir': predicted_dir,
                'actual_ret': actual_ret,
                'actual_dir': actual_dir,
                'is_correct': (predicted_dir == actual_dir) and (predicted_dir != 0),
                'entry_price': entry_price,
                'exit_price': exit_price
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
    <tr><th>Date</th><th>Pred</th><th>Entry</th><th>Exit</th><th>Actual Ret</th><th>Outcome</th><th>PnL</th></tr>
    """
    for _, row in active.tail(50).iterrows():
        p_str = "UP" if row['predicted_dir'] > 0 else "DOWN"
        color = "green" if row['is_correct'] else "red"
        table_html += f"<tr><td>{row['timestamp']}</td><td>{p_str}</td><td>{row['entry_price']:.2f}</td><td>{row['exit_price']:.2f}</td><td>{row['actual_ret']:.4f}</td><td style='color:{color}'>{row['is_correct']}</td><td>{row['pnl']:.4f}</td></tr>"
    table_html += "</table>"
    
    html_out = f"<h3>Accuracy: {accuracy:.2f}% ({correct}/{total})</h3><img src='data:image/png;base64,{plot_url}'/><br>{table_html}"

    # --- API Data Generation ---
    # Convert timestamps to string for JSON serialization
    equity_curve = active[['timestamp', 'cum_pnl']].copy()
    equity_curve['timestamp'] = equity_curve['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Create full trade history for API
    trade_history = active[['timestamp', 'predicted_dir', 'entry_price', 'exit_price', 'actual_ret', 'is_correct', 'pnl']].copy()
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
                
                if len(df_derived) >= 2:
                    actual_close_ret = df_derived.iloc[-2]['close_ret']
                    actual_dir = 1 if actual_close_ret > 0 else -1
                    if actual_close_ret == 0: actual_dir = 0
                    
                    exit_price = df_live.iloc[-2]['close']
                    entry_price = last_pred.get('entry_price', exit_price)
                    raw_return = (exit_price - entry_price) / entry_price if entry_price > 0 else 0
                    
                    if last_pred['pred_dir'] == 0:
                        last_pred['outcome'] = "Flat"
                        last_pred['pnl'] = 0.0
                    else:
                        last_pred['outcome'] = (last_pred['pred_dir'] == actual_dir)
                        last_pred['pnl'] = last_pred['pred_dir'] * raw_return

                    last_pred['actual_ret'] = actual_close_ret
                    last_pred['exit_price'] = exit_price
            
            # Step 2: Make NEW prediction
            if len(df_derived) >= D:
                recent_context = df_derived.iloc[-D:-1][['open_ret', 'high_ret', 'low_ret', 'close_ret']].values
                recent_context_flat = recent_context.reshape(-1)
                
                model_context = model_sequences[:, :D-1, :]
                model_context_flat = model_context.reshape(model_context.shape[0], -1)
                
                diff = np.abs(model_context_flat - recent_context_flat)
                matches_idx = np.where(np.all(diff < E, axis=1))[0]
                
                entry_price = df_live.iloc[-2]['close']
                
                if len(matches_idx) > 0:
                    model_outcomes = model_sequences[matches_idx, -1, :]
                    avg_ret = np.mean(model_outcomes[:, 3])
                    pred_dir = 1 if avg_ret > 0 else -1
                    if avg_ret == 0: pred_dir = 0
                    
                    live_outcomes.append({
                        'time': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
                        'pred_dir': pred_dir,
                        'matches': int(len(matches_idx)), # ensure int for JSON
                        'entry_price': entry_price
                    })
                else:
                    live_outcomes.append({
                        'time': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
                        'pred_dir': 0,
                        'matches': 0,
                        'note': 'No Match',
                        'entry_price': entry_price
                    })
            
            if len(live_outcomes) > 336:
                live_outcomes.pop(0)
                
        except Exception as e:
            print(f"Live loop error: {e}")
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
        live_html = "<h2>Live Outcomes (Last 2 weeks)</h2><table border='1'><tr><th>Time</th><th>Pred</th><th>Matches</th><th>Entry</th><th>Exit</th><th>Outcome</th><th>PnL</th></tr>"
        for item in reversed(live_outcomes):
            outcome_str = item.get('outcome', 'Pending...')
            pnl_str = f"{item.get('pnl', 0):.4f}" if 'pnl' in item else "-"
            entry_s = f"{item.get('entry_price', 0):.2f}"
            exit_s = f"{item.get('exit_price', 0):.2f}" if 'exit_price' in item else "-"
            
            pred_s = "UP" if item['pred_dir'] == 1 else ("DOWN" if item['pred_dir'] == -1 else "FLAT")
            live_html += f"<tr><td>{item['time']}</td><td>{pred_s}</td><td>{item['matches']}</td><td>{entry_s}</td><td>{exit_s}</td><td>{outcome_str}</td><td>{pnl_str}</td></tr>"
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
