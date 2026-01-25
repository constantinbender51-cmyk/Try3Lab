import ccxt
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
C_TOP = 0.20
D_LEN = 4 
E_SIM = 10000
API_PORT = 8080

# Global state
live_log = []
current_prediction = {}
backtest_results = []
recent_perf_results = []
model_patterns_data = [] # Stores full dict with scores for display
model_patterns_seqs = [] # Stores just numpy arrays for logic
debug_logs = [] # Stores first 3 iterations of matching logic
raw_plot_url = ""
derived_plot_url = ""
app = Flask(__name__)

# --- Functions ---

def fetch(timeframe, symbol, start, end):
    print(f"Fetching data for {symbol} ({timeframe})...")
    exchange = ccxt.binance({'enableRateLimit': True})
    start_ts = exchange.parse8601(start) if start else None
    end_ts = exchange.parse8601(end) if end else exchange.milliseconds()
    
    all_ohlcv = []
    since = start_ts
    
    while since < end_ts:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit=1000)
            if not ohlcv: break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            time.sleep(0.1)
        except Exception as e:
            print(f"Error fetching: {e}")
            break
            
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    if end: df = df[df['timestamp'] <= end_ts]
    return df

def deriveround(df):
    df_derived = df.copy()
    cols = ['open', 'high', 'low', 'close']
    derived_cols = []
    for col in cols:
        df_derived[f'{col}_pct'] = df[col].pct_change()
        derived_cols.append(f'{col}_pct')
    df_derived[derived_cols] = df_derived[derived_cols].fillna(0)
    return df_derived

def split(df, b):
    split_idx = int(len(df) * b)
    return df.iloc[:split_idx].reset_index(drop=True), df.iloc[split_idx:].reset_index(drop=True)

def check_similarity(seq1, seq2, threshold):
    v1 = seq1.flatten()
    v2 = seq2.flatten()
    with np.errstate(divide='ignore', invalid='ignore'):
        diff = np.abs((v2 - v1) / v1)
        mask_zeros = (np.abs(v1) < 1e-9)
        if np.any(mask_zeros):
            diff[mask_zeros] = np.abs(v2[mask_zeros])
    return not np.any(diff >= threshold)

def gettop(train_df, c, d, e):
    print("Mining top patterns...")
    cols = ['open_pct', 'high_pct', 'low_pct', 'close_pct']
    data = train_df[cols].values
    
    sequences = []
    for i in range(len(data) - d + 1):
        sequences.append(data[i : i+d])
    
    if not sequences: return []

    seq_arr = np.array(sequences)
    n = len(seq_arr)
    scores = np.zeros(n, dtype=int)
    
    # Optimization: limit to last 2000 for display/speed if dataset is huge, 
    # but strict instruction implies iterating Split 1. 
    # Warning: O(N^2) is slow.
    limit_n = min(n, 2000) # Safety limit for this demo script execution
    print(f"Scoring last {limit_n} sequences in training set...")
    
    start_idx = n - limit_n
    
    for i in range(start_idx, n):
        if i % 100 == 0: print(f"Scoring {i}/{n}", end='\r')
        count = 0
        target = seq_arr[i]
        # Compare against all others in the LIMITED window to keep script responsive
        for j in range(start_idx, n):
            if i == j: continue 
            if check_similarity(target, seq_arr[j], e):
                count += 1
        scores[i] = count
    print("")

    scored_patterns = []
    for i in range(start_idx, n):
        scored_patterns.append({'sequence': seq_arr[i], 'score': scores[i]})
        
    scored_patterns.sort(key=lambda x: x['score'], reverse=True)
    
    top_n = int(len(scored_patterns) * c)
    if top_n < 1: top_n = 1
    
    return scored_patterns[:top_n]

def completesimilarbeginnings(target_df, model_patterns_seqs, d, e, raw_df=None, collect_debug=False):
    results = []
    cols = ['open_pct', 'high_pct', 'low_pct', 'close_pct']
    data = target_df[cols].values
    beg_len = d - 1
    
    global debug_logs
    if collect_debug:
        debug_logs = []

    # Iterating through target data
    for i in range(len(data) - beg_len): 
        current_seq = data[i : i + beg_len]
        prediction = 0 
        match_found = False
        
        # DEBUG: Log first 3 iterations only
        is_debug_iter = collect_debug and (len(debug_logs) < 3)
        iter_log = {'iteration': i, 'input': current_seq.tolist(), 'matches': []} if is_debug_iter else None

        for idx, pattern in enumerate(model_patterns_seqs):
            pattern_beg = pattern[:beg_len]
            is_match = check_similarity(current_seq, pattern_beg, e)
            
            if is_debug_iter:
                # Log the first 3 patterns checked per iteration to avoid huge logs
                if len(iter_log['matches']) < 3:
                    iter_log['matches'].append({
                        'pattern_idx': idx,
                        'pattern_beg': pattern_beg.tolist(),
                        'match': is_match
                    })

            if is_match:
                outcome_candle = pattern[-1]
                close_change = outcome_candle[3]
                if close_change > 0: prediction = 1
                elif close_change < 0: prediction = -1
                match_found = True
                if is_debug_iter:
                    iter_log['prediction'] = prediction
                    iter_log['found_match'] = True
                break 
        
        if is_debug_iter:
            if 'found_match' not in iter_log: iter_log['found_match'] = False
            debug_logs.append(iter_log)

        if match_found and prediction != 0:
            outcome_idx = i + beg_len
            if outcome_idx < len(data):
                entry_price = 0
                exit_price = 0
                raw_input_candles = []
                
                if raw_df is not None:
                    raw_input_candles = raw_df.iloc[i : i + beg_len][['open','close']].values.tolist()
                    entry_price = raw_df.iloc[outcome_idx]['open']
                    exit_price = raw_df.iloc[outcome_idx]['close']
                
                pnl = 0
                if entry_price > 0:
                    pnl = prediction * (exit_price - entry_price) / entry_price
                
                results.append({
                    'timestamp': target_df.iloc[outcome_idx]['datetime'],
                    'input_candles': raw_input_candles,
                    'prediction': "LONG" if prediction == 1 else "SHORT",
                    'entry': entry_price,
                    'exit': exit_price,
                    'outcome': "WIN" if pnl > 0 else ("LOSS" if pnl < 0 else "FLAT"),
                    'pnl': pnl
                })
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
    # Plot 1: Raw DF (Close Prices)
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

    # Plot 2: Derived DF (Close % Change)
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
            # Calculate wait time...
            tf_seconds = 0
            if 'm' in TIMEFRAME: tf_seconds = int(TIMEFRAME.replace('m','')) * 60
            elif 'h' in TIMEFRAME: tf_seconds = int(TIMEFRAME.replace('h','')) * 3600
            
            timestamp = now.timestamp()
            next_ts = (int(timestamp / tf_seconds) + 1) * tf_seconds
            wait_seconds = next_ts - timestamp + 5 
            if wait_seconds < 0: wait_seconds = 0
            time.sleep(wait_seconds)
            
            raw_df = fetch(TIMEFRAME, SYMBOL, None, None)
            derived_df = deriveround(raw_df)
            input_seq_df = derived_df.iloc[-(D_LEN): -1] 
            cols = ['open_pct', 'high_pct', 'low_pct', 'close_pct']
            seq_values = input_seq_df[cols].values
            
            pred_dir = 0
            for pattern in model_patterns_seqs:
                pattern_beg = pattern[:D_LEN-1]
                if check_similarity(seq_values, pattern_beg, E_SIM):
                    if pattern[-1][3] > 0: pred_dir = 1
                    elif pattern[-1][3] < 0: pred_dir = -1
                    break
            
            entry_price = raw_df.iloc[-1]['open']
            pred_obj = {
                'timestamp': datetime.utcnow(),
                'input_candles': raw_df.iloc[-(D_LEN): -1][['open','close']].values.tolist(),
                'prediction': "LONG" if pred_dir == 1 else ("SHORT" if pred_dir == -1 else "FLAT"),
                'entry': entry_price,
                'status': 'OPEN',
                'exit': None, 'pnl': 0
            }
            current_prediction = pred_obj
            
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
            
        except Exception as e:
            print(f"Error in live loop: {e}")
            time.sleep(60)

# --- Server ---

@app.route('/')
def dashboard():
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
    
    # Format Top 10 patterns for display
    top_patterns_display = model_patterns_data[:10]
    
    html = f"""
    <html>
    <head><title>Trading Bot</title>
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
        <h1>Bot Status: {SYMBOL} {TIMEFRAME}</h1>
        <div>
            <b>Backtest Accuracy:</b> {acc:.2%} ({total} trades)<br>
            <b>Current Prediction:</b> {current_prediction.get('prediction','N/A')} @ {current_prediction.get('entry',0)}
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

        <h2>2. Top 10 Model Patterns (Found {len(model_patterns_data)})</h2>
        <table>
            <tr><th>Rank</th><th>Score (Freq)</th><th>Sequence Data (O,H,L,C pct) - Last Candle is Target</th></tr>
            {''.join([f"<tr><td>{i+1}</td><td>{p['score']}</td><td><div class='code-block'>{p['sequence'].tolist()}</div></td></tr>" for i, p in enumerate(top_patterns_display)])}
        </table>

        <h2>3. Algorithm Logic Debug (First 3 Iterations of Backtest)</h2>
        <div class="code-block">
        <pre>
{''.join([f"ITERATION {l['iteration']}:\\n  Input Sequence: {l['input']}\\n  Match Found: {l['found_match']}\\n  Checked Patterns (First 3): {l['matches']}\\n\\n" for l in debug_logs])}
        </pre>
        </div>

        <h2>4. Performance Metrics</h2>
        <img src="data:image/png;base64,{backtest_plot_url}" style="width:100%; max-width:1000px">
        
        <h3>Recent Performance (14d)</h3>
        <table>
            <tr><th>Time</th><th>Pred</th><th>Entry</th><th>Exit</th><th>Outcome</th><th>PnL</th></tr>
             {''.join([f"<tr><td>{r['timestamp']}</td><td>{r['prediction']}</td><td>{r['entry']}</td><td>{r['exit']}</td><td class='{r['outcome'].lower()}'>{r['outcome']}</td><td>{r['pnl']:.4f}</td></tr>" for r in reversed(recent_perf_results)])}
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
        'top_patterns': [{'score': int(x['score']), 'seq': x['sequence'].tolist()} for x in model_patterns_data[:10]],
        'debug_logs': debug_logs
    })

def main():
    global model_patterns_data, model_patterns_seqs, backtest_results, recent_perf_results
    global raw_plot_url, derived_plot_url
    
    df = fetch(TIMEFRAME, SYMBOL, START_STR, END_STR)
    df_derived = deriveround(df)
    
    # Generate static plots for raw and derived
    raw_plot_url, derived_plot_url = generate_static_plots(df, df_derived)
    
    train_df, test_df = split(df_derived, B_SPLIT)
    train_raw, test_raw = split(df, B_SPLIT)
    
    # Get patterns with scores
    model_patterns_data = gettop(train_df, C_TOP, D_LEN, E_SIM)
    # Extract just sequences for logic
    model_patterns_seqs = [x['sequence'] for x in model_patterns_data]
    
    print("Running Backtest...")
    # Enable debug collection for backtest
    backtest_results = completesimilarbeginnings(test_df, model_patterns_seqs, D_LEN, E_SIM, test_raw, collect_debug=True)
    
    print("Calculating Recent Performance...")
    two_weeks_ago = datetime.utcnow() - timedelta(days=14)
    mask = df['datetime'] > two_weeks_ago
    recent_perf_results = completesimilarbeginnings(df_derived[mask], model_patterns_seqs, D_LEN, E_SIM, df[mask])
    
    acc, _, _ = get_accuracy_metrics(backtest_results)
    print(f"Backtest Accuracy: {acc:.2%}")
    
    t = threading.Thread(target=live_loop_thread)
    t.daemon = True
    t.start()
    
    print(f"Serving on port {API_PORT}")
    app.run(host='0.0.0.0', port=API_PORT)

if __name__ == '__main__':
    main()
