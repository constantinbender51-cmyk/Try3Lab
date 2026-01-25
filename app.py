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
START_STR = '2023-01-01 00:00:00' 
END_STR = None 
B_SPLIT = 0.70
C_TOP = 0.1
D_LEN = 3
E_SIM = 1 # Note: The original 1 (100%) is very loose; usually 0.05-0.1 is standard, but kept as requested.
API_PORT = 8080

# Global state
live_log = []
current_prediction = {}
backtest_results = []
recent_perf_results = []
model_patterns_seqs = np.array([]) # Storing as numpy array for speed
model_patterns_meta = [] # Storing scores/indices
debug_logs = [] 
raw_plot_url = ""
derived_plot_url = ""
plot_lock = threading.Lock()
app = Flask(__name__)

# --- Functions ---

def fetch(timeframe, symbol, start_str, end_str):
    """
    Fetches OHLCV data using requests from Binance API v3.
    """
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
        params = {
            'symbol': api_symbol,
            'interval': timeframe,
            'startTime': current_start,
            'limit': 1000
        }
        
        try:
            response = requests.get(base_url, params=params)
            data = response.json()
            
            if isinstance(data, dict) and 'code' in data:
                print(f"API Error: {data}")
                break
            
            if not data:
                break
                
            all_data.extend(data)
            
            # Binance klines: [Open Time, Open, High, Low, Close, Vol, Close Time, ...]
            # Use Close Time + 1ms for next start
            current_start = data[-1][6] + 1
            
            if data[-1][0] >= end_ts:
                break
                
            time.sleep(0.05) 
            
        except Exception as e:
            print(f"Error fetching: {e}")
            break
    
    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 
        'close_time', 'quote_asset_vol', 'trades', 'tb_base_vol', 'tb_quote_vol', 'ignore'
    ])
    
    cols = ['open', 'high', 'low', 'close', 'volume']
    df[cols] = df[cols].apply(pd.to_numeric, axis=1)
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    if end_str:
        df = df[df['timestamp'] <= end_ts]
        
    return df

def deriveround(df):
    df_derived = df.copy()
    cols = ['open', 'high', 'low', 'close']
    derived_cols = []
    for col in cols:
        df_derived[f'{col}_pct'] = df[col].pct_change()
        derived_cols.append(f'{col}_pct')
    
    # Replace infinite values with 0 and fill NaNs
    df_derived[derived_cols] = df_derived[derived_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    return df_derived

def split(df, b):
    if len(df) == 0: return df, df
    split_idx = int(len(df) * b)
    return df.iloc[:split_idx].reset_index(drop=True), df.iloc[split_idx:].reset_index(drop=True)

def create_sequences(data, d):
    """
    Vectorized sequence creation.
    Returns array of shape (N, d, 4)
    """
    n_samples = len(data) - d + 1
    if n_samples <= 0:
        return np.array([])
    
    # Using stride_tricks to create a sliding window view (memory efficient)
    # data shape: (Rows, Features)
    # Result shape: (n_samples, d, Features)
    window_shape = (n_samples, d, data.shape[1])
    window_strides = (data.strides[0], data.strides[0], data.strides[1])
    
    sequences = np.lib.stride_tricks.as_strided(
        data, shape=window_shape, strides=window_strides, writeable=False
    )
    return sequences.copy() # Return a copy to ensure memory safety in threading

def gettop(train_df, c, d, e):
    print("Mining top patterns (Vectorized)...")
    cols = ['open_pct', 'high_pct', 'low_pct', 'close_pct']
    # Convert to float32 for speed/memory if dataset is large
    data_values = train_df[cols].values.astype(np.float32)
    
    # Create all sequences: Shape (N, D, 4)
    seq_arr = create_sequences(data_values, d)
    n = len(seq_arr)
    if n == 0: return [], np.array([])
    
    # Limit scoring to recent history to save time, as per original logic
    limit_n = min(n, 2000) 
    start_idx = n - limit_n
    
    scores = np.zeros(limit_n, dtype=int)
    
    print(f"Scoring last {limit_n} sequences against {n} historical sequences...")
    
    # --- Vectorized Scoring Loop ---
    # Instead of Python loop matching seq vs seq, we compare 
    # one sequence against the entire matrix at once.
    
    # Flatten sequences for easier comparison: (N, D*4)
    flat_seqs = seq_arr.reshape(n, -1)
    
    # We only score the target slice
    target_slice = flat_seqs[start_idx:]
    
    # Iterate through the targets we want to score
    for i in range(limit_n):
        if i % 100 == 0: print(f"Scoring {i}/{limit_n}", end='\r')
        
        target_vec = target_slice[i] # Shape (features,)
        
        # Calculate Percentage Difference Vectorized
        # diff = |(matrix - target) / target|
        # Handle division by zero carefully
        
        with np.errstate(divide='ignore', invalid='ignore'):
            # This creates a boolean matrix where match is True
            # We want rows where ALL features match
            
            # Using a small epsilon for zero division protection
            safe_target = np.where(np.abs(target_vec) < 1e-9, 1e-9, target_vec)
            diffs = np.abs((flat_seqs - target_vec) / safe_target)
            
            # If target was effectively zero, check absolute difference instead of relative
            # (Hybrid approach for stability)
            zero_mask = (np.abs(target_vec) < 1e-9)
            if np.any(zero_mask):
                diffs[:, zero_mask] = np.abs(flat_seqs[:, zero_mask])
            
            # Check threshold
            matches = np.all(diffs < e, axis=1)
            
            # Count True values (subtract 1 because it matches itself)
            scores[i] = np.sum(matches) - 1
            if scores[i] < 0: scores[i] = 0

    print("\nScoring complete.")

    # Package results
    scored_patterns = []
    for i in range(limit_n):
        idx_in_full = start_idx + i
        scored_patterns.append({
            'sequence': seq_arr[idx_in_full], 
            'score': scores[i],
            'orig_idx': idx_in_full
        })
        
    scored_patterns.sort(key=lambda x: x['score'], reverse=True)
    
    top_n = int(len(scored_patterns) * c)
    if top_n < 1: top_n = 1
    
    top_items = scored_patterns[:top_n]
    
    # Return metadata list and a clean numpy array of just the sequences for fast inference
    top_seqs_array = np.array([p['sequence'] for p in top_items])
    
    return top_items, top_seqs_array

def completesimilarbeginnings(target_df, model_patterns_arr, d, e, raw_df=None, collect_debug=False):
    """
    Vectorized Backtesting/Inference.
    model_patterns_arr shape: (NumPatterns, D, 4)
    """
    if len(model_patterns_arr) == 0: return []
    
    results = []
    cols = ['open_pct', 'high_pct', 'low_pct', 'close_pct']
    data_values = target_df[cols].values.astype(np.float32)
    
    # We look at D-1 length for matching
    beg_len = d - 1
    
    # Create input sequences from target data: Shape (M, beg_len, 4)
    target_seqs = create_sequences(data_values, beg_len)
    
    # Flatten inputs: (M, beg_len*4)
    flat_targets = target_seqs.reshape(len(target_seqs), -1)
    
    # Prepare Model Patterns: Take first D-1 rows, flatten: (NumPatterns, beg_len*4)
    model_begs = model_patterns_arr[:, :beg_len, :].reshape(len(model_patterns_arr), -1)
    
    # Extract Model Outcomes (Last candle, close_pct column is index 3): Shape (NumPatterns,)
    model_outcomes = model_patterns_arr[:, -1, 3]
    
    global debug_logs
    if collect_debug:
        debug_logs = []

    # Iterate through validation/test time steps
    # Note: We iterate because for each step we need to check against ALL models.
    # While we could fully matrixize (M x P), memory might explode if M and P are large.
    # Row-by-row against Matrix is a good balance.
    
    num_steps = len(flat_targets)
    
    for i in range(num_steps):
        target_vec = flat_targets[i]
        
        # --- Vectorized Match ---
        with np.errstate(divide='ignore', invalid='ignore'):
            safe_target = np.where(np.abs(target_vec) < 1e-9, 1e-9, target_vec)
            diffs = np.abs((model_begs - target_vec) / safe_target)
            
            zero_mask = (np.abs(target_vec) < 1e-9)
            if np.any(zero_mask):
                diffs[:, zero_mask] = np.abs(model_begs[:, zero_mask])
                
            matches = np.all(diffs < e, axis=1) # Boolean array of shape (NumPatterns,)
        
        # Check if any match found
        match_indices = np.where(matches)[0]
        
        prediction = 0
        
        # Debug Logging
        if collect_debug and len(debug_logs) < 3:
            iter_log = {
                'iteration': i, 
                'input': target_seqs[i].tolist(), 
                'matches': [],
                'found_match': False
            }
            # Log first few potential matches
            for m_idx in match_indices[:3]:
                iter_log['matches'].append({
                    'pattern_idx': int(m_idx),
                    'match': True
                })
            
            if len(match_indices) > 0:
                iter_log['found_match'] = True
            debug_logs.append(iter_log)

        if len(match_indices) > 0:
            # Found matches. 
            # Strategy: Take the first match (as per original code logic "break")
            # Or could vote. Original code used "break", so we take index 0.
            first_match_idx = match_indices[0]
            close_change = model_outcomes[first_match_idx]
            
            if close_change > 0: prediction = 1
            elif close_change < 0: prediction = -1
        
        if prediction != 0:
            outcome_idx = i + beg_len
            # Check bounds
            if outcome_idx < len(target_df):
                entry_price = 0
                exit_price = 0
                raw_input_candles = []
                
                if raw_df is not None:
                    # Adjust index mapping between derived (shorter) and raw
                    # derived starts at index 1 of raw usually, but here mapped by date
                    # Safest is to use the outcome_idx directly if aligned
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
            
            raw_df = fetch(TIMEFRAME, SYMBOL, None, None)
            if raw_df.empty: continue
            
            derived_df = deriveround(raw_df)
            
            # Prepare Input
            input_seq_df = derived_df.iloc[-(D_LEN): -1] 
            cols = ['open_pct', 'high_pct', 'low_pct', 'close_pct']
            
            # Shape (D-1, 4)
            current_seq = input_seq_df[cols].values.astype(np.float32)
            flat_current = current_seq.flatten() # (Features,)
            
            pred_dir = 0
            
            # Vectorized Match against Model Patterns
            if len(model_patterns_seqs) > 0:
                # Prepare models: (N, D-1, 4) -> Flatten -> (N, Features)
                beg_len = D_LEN - 1
                model_begs = model_patterns_seqs[:, :beg_len, :].reshape(len(model_patterns_seqs), -1)
                
                with np.errstate(divide='ignore', invalid='ignore'):
                    safe_target = np.where(np.abs(flat_current) < 1e-9, 1e-9, flat_current)
                    diffs = np.abs((model_begs - flat_current) / safe_target)
                    
                    zero_mask = (np.abs(flat_current) < 1e-9)
                    if np.any(zero_mask):
                        diffs[:, zero_mask] = np.abs(model_begs[:, zero_mask])
                    
                    matches = np.all(diffs < E_SIM, axis=1)
                
                match_indices = np.where(matches)[0]
                if len(match_indices) > 0:
                    # Take first match logic
                    idx = match_indices[0]
                    # close_pct is index 3
                    outcome = model_patterns_seqs[idx, -1, 3]
                    if outcome > 0: pred_dir = 1
                    elif outcome < 0: pred_dir = -1

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
            
            # Close previous prediction
            if len(live_log) > 0 and live_log[-1]['status'] == 'OPEN':
                last = live_log[-1]
                # Assuming the candle that just closed is the outcome for the previous prediction
                exit_price = raw_df.iloc[-2]['close'] 
                last['exit'] = exit_price
                d_val = 1 if last['prediction'] == "LONG" else (-1 if last['prediction'] == "SHORT" else 0)
                if d_val != 0:
                    last['pnl'] = d_val * (exit_price - last['entry']) / last['entry']
                    last['outcome'] = "WIN" if last['pnl'] > 0 else "LOSS"
                else: last['outcome'] = "FLAT"
                last['status'] = 'CLOSED'
            
            if pred_dir != 0: live_log.append(pred_obj)
            
            # Prune log
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
    top_patterns_display = model_patterns_meta[:10]
    
    html = f"""
    <html>
    <head><title>Trading Bot (Optimized)</title>
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
        <h1>Bot Status: {SYMBOL} {TIMEFRAME} (Vectorized)</h1>
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

        <h2>2. Top 10 Model Patterns (Found {len(model_patterns_meta)})</h2>
        <table>
            <tr><th>Rank</th><th>Score (Freq)</th><th>Sequence Data</th></tr>
            {''.join([f"<tr><td>{i+1}</td><td>{p['score']}</td><td><div class='code-block'>{p['sequence'].tolist()}</div></td></tr>" for i, p in enumerate(top_patterns_display)])}
        </table>

        <h2>3. Algorithm Logic Debug (First 3 Iterations of Backtest)</h2>
        <div class="code-block">
        <pre>
{''.join([f"ITERATION {l['iteration']}:\\n  Match: {l['found_match']}\\n  Checks: {l['matches']}\\n\\n" for l in debug_logs])}
        </pre>
        </div>

        <h2>4. Live Prediction Log</h2>
        <table>
            <tr><th>Time</th><th>Pred</th><th>Input Candles (O,C)</th><th>Entry</th><th>Exit</th><th>Outcome</th><th>PnL</th></tr>
            {''.join([f"<tr><td>{r['timestamp']}</td><td>{r['prediction']}</td><td><div class='code-block'>{r['input_candles']}</div></td><td>{r['entry']}</td><td>{r['exit']}</td><td class='{r.get('outcome','').lower()}'>{r.get('outcome','OPEN')}</td><td>{r['pnl']:.4f}</td></tr>" for r in reversed(live_log)])}
        </table>

        <h2>5. Recent Performance (Last 14d)</h2>
        <img src="data:image/png;base64,{backtest_plot_url}" style="width:100%; max-width:1000px">
        <table>
            <tr><th>Time</th><th>Pred</th><th>Input Candles (O, C)</th><th>Entry</th><th>Exit</th><th>Outcome</th><th>PnL</th></tr>
             {''.join([f"<tr><td>{r['timestamp']}</td><td>{r['prediction']}</td><td><div class='code-block'>{r['input_candles']}</div></td><td>{r['entry']}</td><td>{r['exit']}</td><td class='{r['outcome'].lower()}'>{r['outcome']}</td><td>{r['pnl']:.4f}</td></tr>" for r in reversed(recent_perf_results)])}
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
        'top_patterns': [{'score': int(x['score']), 'seq': x['sequence'].tolist()} for x in model_patterns_meta[:10]],
        'debug_logs': debug_logs
    })

def main():
    global model_patterns_meta, model_patterns_seqs, backtest_results, recent_perf_results
    global raw_plot_url, derived_plot_url
    
    # 1. Fetch
    df = fetch(TIMEFRAME, SYMBOL, START_STR, END_STR)
    if df.empty:
        print("No data fetched! Check dates or network.")
        return
    print(f"Data fetched: {len(df)} rows.")

    # 2. Derive
    df_derived = deriveround(df)
    
    # 3. Plot (One time)
    raw_plot_url, derived_plot_url = generate_static_plots(df, df_derived)
    
    # 4. Split
    train_df, test_df = split(df_derived, B_SPLIT)
    train_raw, test_raw = split(df, B_SPLIT)
    
    # 5. Train (Pattern Mining)
    # Returns metadata list and pure numpy array for inference
    model_patterns_meta, model_patterns_seqs = gettop(train_df, C_TOP, D_LEN, E_SIM)
    
    # 6. Backtest
    print("Running Backtest...")
    backtest_results = completesimilarbeginnings(test_df, model_patterns_seqs, D_LEN, E_SIM, test_raw, collect_debug=True)
    
    # 7. Recent Performance
    print("Calculating Recent Performance...")
    two_weeks_ago = datetime.utcnow() - timedelta(days=14)
    if df['datetime'].max() > two_weeks_ago:
        mask = df['datetime'] > two_weeks_ago
        recent_perf_results = completesimilarbeginnings(df_derived[mask], model_patterns_seqs, D_LEN, E_SIM, df[mask])
    else:
        recent_perf_results = []
    
    acc, _, _ = get_accuracy_metrics(backtest_results)
    print(f"Backtest Accuracy: {acc:.2%}")
    
    # 8. Start Live Loop
    t = threading.Thread(target=live_loop_thread)
    t.daemon = True
    t.start()
    
    print(f"Serving on port {API_PORT}")
    app.run(host='0.0.0.0', port=API_PORT)

if __name__ == '__main__':
    main()
