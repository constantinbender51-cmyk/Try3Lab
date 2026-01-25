import requests
import pandas as pd
import numpy as np
import time
import threading
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, jsonify, render_template_string
from datetime import datetime, timedelta

# --- Parameters ---
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1h'
START_STR = '2026-01-01 00:00:00' 
END_STR = None 
B_SPLIT = 0.70
C_TOP = 0.20 # Unused now
D_LEN = 4 
E_SIM = 0.1 # Adjusted for Sum of Absolute Differences (SAD). 0.5 is a reasonable cumulative error threshold for pct changes.
API_PORT = 8080

# Global state
live_log = []
current_prediction = {}
backtest_results = []
recent_perf_results = []
train_sequences = np.array([]) # The "Library" of history (Split 1)
debug_logs = [] 
raw_plot_url = ""
derived_plot_url = ""
plot_lock = threading.Lock()
app = Flask(__name__)

# --- Functions ---

def fetch(timeframe, symbol, start_str, end_str):
    print(f"Fetching data for {symbol} ({timeframe}) via requests...")
    api_symbol = symbol.replace('/', '')
    base_url = 'https://api.binance.com/api/v3/klines'
    
    if start_str:
        dt_obj = datetime.strptime(start_str, '%Y-%m-%d %H:%M:%S')
        start_ts = int(dt_obj.timestamp() * 1000)
    else:
        start_ts = int((datetime.now() - timedelta(days=1)).timestamp() * 1000)
        
    if end_str:
        dt_end = datetime.strptime(end_str, '%Y-%m-%d %H:%M:%S')
        end_ts = int(dt_end.timestamp() * 1000)
    else:
        end_ts = int(time.time() * 1000)
        
    all_data = []
    current_start = start_ts
    
    while current_start < end_ts:
        params = {'symbol': api_symbol, 'interval': timeframe, 'startTime': current_start, 'limit': 1000}
        try:
            response = requests.get(base_url, params=params)
            data = response.json()
            if isinstance(data, dict) and 'code' in data: break
            if not data: break
            all_data.extend(data)
            current_start = data[-1][6] + 1
            if data[-1][0] >= end_ts: break
            time.sleep(0.05) 
        except Exception as e:
            print(f"Error fetching: {e}")
            break
    
    if not all_data: return pd.DataFrame()

    df = pd.DataFrame(all_data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 
        'close_time', 'quote_asset_vol', 'trades', 'tb_base_vol', 'tb_quote_vol', 'ignore'
    ])
    
    cols = ['open', 'high', 'low', 'close', 'volume']
    df[cols] = df[cols].apply(pd.to_numeric, axis=1)
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    if end_str: df = df[df['timestamp'] <= end_ts]
    return df

def deriveround(df):
    df_derived = df.copy()
    cols = ['open', 'high', 'low', 'close']
    derived_cols = []
    for col in cols:
        df_derived[f'{col}_pct'] = df[col].pct_change()
        derived_cols.append(f'{col}_pct')
    
    df_derived[derived_cols] = df_derived[derived_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    return df_derived

def split(df, b):
    if len(df) == 0: return df, df
    split_idx = int(len(df) * b)
    return df.iloc[:split_idx].reset_index(drop=True), df.iloc[split_idx:].reset_index(drop=True)

def create_sequences(data, d):
    n_samples = len(data) - d + 1
    if n_samples <= 0: return np.array([])
    window_shape = (n_samples, d, data.shape[1])
    window_strides = (data.strides[0], data.strides[0], data.strides[1])
    sequences = np.lib.stride_tricks.as_strided(data, shape=window_shape, strides=window_strides, writeable=False)
    return sequences.copy()

def find_best_match_vote(target_seq, history_seqs, e_threshold):
    """
    1. Calculates Sum of Absolute Differences (SAD) between target and all history.
    2. Filters those < e_threshold.
    3. Votes on outcome (Next Candle Close Change) based on matches.
    """
    if len(history_seqs) == 0: return 0, 0, 0

    # History Seqs shape: (N, D, 4)
    # Target Seq shape: (D-1, 4) (Needs to be compared to first D-1 of history)
    
    beg_len = len(target_seq)
    history_begs = history_seqs[:, :beg_len, :] # (N, beg_len, 4)
    
    # --- 1. Calculate SAD (Sum of Absolute Differences) ---
    # target_seq must be broadcasted
    # diffs shape: (N, beg_len, 4)
    diffs = np.abs(history_begs - target_seq)
    
    # Sum over length and features to get one score per history item
    # axis=(1,2) sums the inner window and features
    scores = np.sum(diffs, axis=(1, 2)) # Shape (N,)
    
    # --- 2. Filter ---
    match_mask = scores < e_threshold
    match_indices = np.where(match_mask)[0]
    
    if len(match_indices) == 0:
        return 0, 0, 0 # No matches found
    
    # --- 3. Vote on Outcome ---
    # Get outcomes of matching sequences (Last candle, Close% is index 3)
    outcomes = history_seqs[match_indices, -1, 3]
    
    longs = np.sum(outcomes > 0)
    shorts = np.sum(outcomes < 0)
    
    # Calculate probability of the dominant direction
    total_matches = len(outcomes)
    
    if longs > shorts:
        prob = longs / total_matches
        return 1, prob, total_matches
    elif shorts > longs:
        prob = shorts / total_matches
        return -1, prob, total_matches
    else:
        return 0, 0, total_matches

def backtest_logic(target_df, training_library, d, e, raw_df=None, collect_debug=False):
    results = []
    cols = ['open_pct', 'high_pct', 'low_pct', 'close_pct']
    data_values = target_df[cols].values.astype(np.float32)
    beg_len = d - 1
    
    # Create windows for the target dataset
    target_seqs = create_sequences(data_values, beg_len) # Shape (M, beg_len, 4)
    
    global debug_logs
    if collect_debug: debug_logs = []

    print(f"Backtesting {len(target_seqs)} candles against {len(training_library)} history patterns...")

    for i in range(len(target_seqs)):
        if i % 100 == 0: print(f"Step {i}/{len(target_seqs)}", end='\r')
        
        current_seq = target_seqs[i]
        
        # Perform Search & Vote
        direction, probability, match_count = find_best_match_vote(current_seq, training_library, e)
        
        # Debugging
        if collect_debug and len(debug_logs) < 3:
            debug_logs.append({
                'iteration': i,
                'input': current_seq.tolist(),
                'found_matches': int(match_count),
                'vote_direction': direction,
                'probability': float(probability)
            })

        if direction != 0:
            outcome_idx = i + beg_len
            if outcome_idx < len(target_df):
                entry_price = 0
                exit_price = 0
                raw_input_candles = []
                
                if raw_df is not None:
                    raw_input_candles = raw_df.iloc[i : i + beg_len][['open','close']].values.tolist()
                    entry_price = raw_df.iloc[outcome_idx]['open']
                    exit_price = raw_df.iloc[outcome_idx]['close']
                
                pnl = 0
                if entry_price > 0:
                    pnl = direction * (exit_price - entry_price) / entry_price
                
                results.append({
                    'timestamp': target_df.iloc[outcome_idx]['datetime'],
                    'input_candles': raw_input_candles,
                    'prediction': "LONG" if direction == 1 else "SHORT",
                    'probability': probability,
                    'matches': match_count,
                    'entry': entry_price,
                    'exit': exit_price,
                    'outcome': "WIN" if pnl > 0 else ("LOSS" if pnl < 0 else "FLAT"),
                    'pnl': pnl
                })
    print("\nBacktest Complete.")
    return results

def get_accuracy_metrics(results):
    if not results: return 0, 0, []
    wins = sum(1 for r in results if r['pnl'] > 0)
    losses = sum(1 for r in results if r['pnl'] < 0)
    total = wins + losses
    acc = (wins / total) if total > 0 else 0
    cum_pnl = np.cumsum([r['pnl'] for r in results])
    return acc, total, cum_pnl

def generate_static_plots(df, df_derived):
    if df.empty: return "", ""
    with plot_lock:
        img = io.BytesIO()
        plt.figure(figsize=(10, 4))
        plt.plot(df['datetime'], df['close'], label='Close Price')
        plt.title(f"Raw Data: {SYMBOL} Close Price")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.tight_layout()
        plt.savefig(img, format='png')
        img.seek(0)
        raw_b64 = base64.b64encode(img.getvalue()).decode()
        plt.close()

        img2 = io.BytesIO()
        plt.figure(figsize=(10, 4))
        plt.plot(df_derived['datetime'], df_derived['close_pct'], color='orange', label='Close % Change', alpha=0.7)
        plt.title(f"Derived Data: {SYMBOL} Close % Change")
        plt.xlabel("Date")
        plt.ylabel("% Change")
        plt.legend()
        plt.tight_layout()
        plt.savefig(img2, format='png')
        img2.seek(0)
        derived_b64 = base64.b64encode(img2.getvalue()).decode()
        plt.close()
    return raw_b64, derived_b64

# --- Threading & Live Logic ---

def live_loop_thread():
    global current_prediction, live_log
    while True:
        try:
            now = datetime.utcnow()
            tf_seconds = 3600
            if 'm' in TIMEFRAME: tf_seconds = int(TIMEFRAME.replace('m','')) * 60
            elif 'h' in TIMEFRAME: tf_seconds = int(TIMEFRAME.replace('h','')) * 3600
            
            timestamp = now.timestamp()
            next_ts = (int(timestamp / tf_seconds) + 1) * tf_seconds
            wait_seconds = next_ts - timestamp + 5 
            if wait_seconds < 0: wait_seconds = 0
            time.sleep(wait_seconds)
            
            # Fetch Live Data
            raw_df = fetch(TIMEFRAME, SYMBOL, None, None)
            if raw_df.empty: continue
            
            derived_df = deriveround(raw_df)
            
            # Get Current Sequence
            input_seq_df = derived_df.iloc[-(D_LEN): -1] 
            cols = ['open_pct', 'high_pct', 'low_pct', 'close_pct']
            current_seq = input_seq_df[cols].values.astype(np.float32) # Shape (D-1, 4)
            
            # Search History (Split 1 Library)
            pred_dir, prob, matches = find_best_match_vote(current_seq, train_sequences, E_SIM)
            
            entry_price = raw_df.iloc[-1]['open']
            pred_obj = {
                'timestamp': datetime.utcnow(),
                'input_candles': raw_df.iloc[-(D_LEN): -1][['open','close']].values.tolist(),
                'prediction': "LONG" if pred_dir == 1 else ("SHORT" if pred_dir == -1 else "FLAT"),
                'probability': float(prob),
                'matches': int(matches),
                'entry': entry_price,
                'status': 'OPEN',
                'exit': None, 'pnl': 0
            }
            current_prediction = pred_obj
            
            # Close previous prediction
            if len(live_log) > 0 and live_log[-1]['status'] == 'OPEN':
                last = live_log[-1]
                exit_price = raw_df.iloc[-2]['close'] 
                last['exit'] = exit_price
                d_val = 1 if last['prediction'] == "LONG" else (-1 if last['prediction'] == "SHORT" else 0)
                if d_val != 0:
                    last['pnl'] = d_val * (exit_price - last['entry']) / last['entry']
                    last['outcome'] = "WIN" if last['pnl'] > 0 else "LOSS"
                else: last['outcome'] = "FLAT"
                last['status'] = 'CLOSED'
            
            if pred_dir != 0: live_log.append(pred_obj)
            
            # Prune
            two_weeks = 14 * 24 * 3600
            now_ts = datetime.utcnow().timestamp()
            live_log[:] = [x for x in live_log if (now_ts - x['timestamp'].timestamp()) < two_weeks]
            
        except Exception as e:
            print(f"Error in live loop: {e}")
            time.sleep(60)

# --- Server ---

@app.route('/')
def dashboard():
    with plot_lock:
        img = io.BytesIO()
        plt.figure(figsize=(12, 5))
        dates = [r['timestamp'] for r in backtest_results]
        pnls = np.cumsum([r['pnl'] for r in backtest_results])
        plt.subplot(1, 2, 1)
        if len(dates) > 0: plt.plot(dates, pnls, label='Backtest PnL')
        plt.title("Backtest PnL")
        
        if recent_perf_results:
            dates_rec = [r['timestamp'] for r in recent_perf_results]
            pnls_rec = np.cumsum([r['pnl'] for r in recent_perf_results])
            plt.subplot(1, 2, 2)
            plt.plot(dates_rec, pnls_rec, color='orange', label='14d Performance')
            plt.title("Recent 14d Performance")
        
        plt.tight_layout()
        plt.savefig(img, format='png')
        img.seek(0)
        backtest_plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
    
    acc, total, _ = get_accuracy_metrics(backtest_results)
    
    html = f"""
    <html>
    <head><title>Trading Bot (Probability Vote)</title>
    <style>
        body {{ font-family: monospace; margin: 20px; }}
        h2 {{ border-bottom: 2px solid #ccc; padding-bottom: 5px; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; font-size: 0.85em; }}
        th, td {{ border: 1px solid #ddd; padding: 6px; text-align: left; }}
        th {{ background-color: #eee; }}
        .win {{ color: green; font-weight: bold; }}
        .loss {{ color: red; font-weight: bold; }}
        .container {{ display: flex; flex-wrap: wrap; }}
        .box {{ flex: 1; min-width: 400px; margin: 10px; }}
        .code-block {{ background: #f4f4f4; padding: 10px; border: 1px solid #ddd; overflow-x: auto; }}
    </style>
    </head>
    <body>
        <h1>Bot Status: {SYMBOL} {TIMEFRAME} (SAD Metric + Voting)</h1>
        <div>
            <b>Backtest Accuracy:</b> {acc:.2%} ({total} trades)<br>
            <b>Current Prediction:</b> {current_prediction.get('prediction','N/A')} (Prob: {current_prediction.get('probability',0):.2f})
        </div>
        
        <h2>1. Data Visualization</h2>
        <div class="container">
            <div class="box">
                <h4>Raw Prices (Close)</h4>
                <img src="data:image/png;base64,{raw_plot_url}" style="width:100%">
            </div>
            <div class="box">
                <h4>Derived Data (Close % Change)</h4>
                <img src="data:image/png;base64,{derived_plot_url}" style="width:100%">
            </div>
        </div>

        <h2>2. Live Prediction Log</h2>
        <table>
            <tr><th>Time</th><th>Pred</th><th>Matches</th><th>Prob</th><th>Entry</th><th>Exit</th><th>Outcome</th><th>PnL</th></tr>
            {''.join([f"<tr><td>{r['timestamp']}</td><td>{r['prediction']}</td><td>{r.get('matches',0)}</td><td>{r.get('probability',0):.2f}</td><td>{r['entry']}</td><td>{r['exit']}</td><td class='{r.get('outcome','').lower()}'>{r.get('outcome','OPEN')}</td><td>{r['pnl']:.4f}</td></tr>" for r in reversed(live_log)])}
        </table>

        <h2>3. Debug (First 3 Backtest Iterations)</h2>
        <div class="code-block">
        <pre>
{''.join([f"ITERATION {l['iteration']}: Matches Found: {l['found_matches']} | Vote: {l['vote_direction']} | Prob: {l['probability']:.2f}\\n" for l in debug_logs])}
        </pre>
        </div>

        <h2>4. Recent Performance (Last 14d)</h2>
        <img src="data:image/png;base64,{backtest_plot_url}" style="width:100%; max-width:1000px">
        <table>
            <tr><th>Time</th><th>Pred</th><th>Matches</th><th>Prob</th><th>Entry</th><th>Exit</th><th>Outcome</th><th>PnL</th></tr>
             {''.join([f"<tr><td>{r['timestamp']}</td><td>{r['prediction']}</td><td>{r['matches']}</td><td>{r['probability']:.2f}</td><td>{r['entry']}</td><td>{r['exit']}</td><td class='{r['outcome'].lower()}'>{r['outcome']}</td><td>{r['pnl']:.4f}</td></tr>" for r in reversed(recent_perf_results)])}
        </table>
    </body>
    </html>
    """
    return render_template_string(html)

@app.route('/api')
def api_endpoint():
    return jsonify({
        'current_prediction': current_prediction,
        'live_log': live_log,
        'debug_logs': debug_logs
    })

def main():
    global train_sequences, backtest_results, recent_perf_results
    global raw_plot_url, derived_plot_url
    
    # 1. Fetch
    df = fetch(TIMEFRAME, SYMBOL, START_STR, END_STR)
    if df.empty:
        print("No data fetched! Check dates or network.")
        return
    print(f"Data fetched: {len(df)} rows.")

    # 2. Derive & Plot
    df_derived = deriveround(df)
    raw_plot_url, derived_plot_url = generate_static_plots(df, df_derived)
    
    # 3. Split
    train_df, test_df = split(df_derived, B_SPLIT)
    train_raw, test_raw = split(df, B_SPLIT)
    
    # 4. Create Library (Split 1)
    print("Building Training Library (Split 1)...")
    cols = ['open_pct', 'high_pct', 'low_pct', 'close_pct']
    train_data = train_df[cols].values.astype(np.float32)
    train_sequences = create_sequences(train_data, D_LEN)
    print(f"Library created with {len(train_sequences)} patterns.")
    
    # 5. Backtest (Split 2 vs Split 1)
    print("Running Backtest...")
    backtest_results = backtest_logic(test_df, train_sequences, D_LEN, E_SIM, test_raw, collect_debug=True)
    
    # 6. Recent Performance
    print("Calculating Recent Performance...")
    two_weeks_ago = datetime.utcnow() - timedelta(days=14)
    if df['datetime'].max() > two_weeks_ago:
        mask = df['datetime'] > two_weeks_ago
        recent_perf_results = backtest_logic(df_derived[mask], train_sequences, D_LEN, E_SIM, df[mask])
    else:
        recent_perf_results = []
    
    acc, _, _ = get_accuracy_metrics(backtest_results)
    print(f"Backtest Accuracy: {acc:.2%}")
    
    # 7. Start Live Loop
    t = threading.Thread(target=live_loop_thread)
    t.daemon = True
    t.start()
    
    print(f"Serving on port {API_PORT}")
    app.run(host='0.0.0.0', port=API_PORT)

if __name__ == '__main__':
    main()
