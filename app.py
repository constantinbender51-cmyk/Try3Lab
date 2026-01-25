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
START_STR = '2023-01-01 00:00:00'
END_STR = None 
B_SPLIT = 0.70  # 70% Training
C_TOP = 0.20  # Top 20% frequent sequences
D_LEN = 4  # Full sequence length
E_SIM = 0.01  # 1% similarity threshold
API_PORT = 8080

# Global state
live_log = []
current_prediction = {}
backtest_results = []
recent_perf_results = []
model_patterns = []
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
    """Calculates pct change; adds 0-row at start."""
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
    """
    Returns True if abs((val2 - val1) / val1) < threshold for all values.
    Handles division by zero by checking absolute difference if val1 is close to 0.
    """
    v1 = seq1.flatten()
    v2 = seq2.flatten()
    
    with np.errstate(divide='ignore', invalid='ignore'):
        # Standard pct diff
        diff = np.abs((v2 - v1) / v1)
        
        # Handle zeros in v1
        mask_zeros = (np.abs(v1) < 1e-9)
        if np.any(mask_zeros):
            # If v1 is 0, we check if v2 is within absolute threshold of 0
            # (treating e as absolute diff for zero-cases)
            diff[mask_zeros] = np.abs(v2[mask_zeros])
            
    return not np.any(diff >= threshold)

def gettop(train_df, c, d, e):
    print("Mining top patterns (Iterating all sequences in Split 1)...")
    cols = ['open_pct', 'high_pct', 'low_pct', 'close_pct']
    data = train_df[cols].values
    
    # 1. Extract all sequences
    sequences = []
    for i in range(len(data) - d + 1):
        sequences.append(data[i : i+d])
    
    if not sequences: return []

    seq_arr = np.array(sequences)
    n = len(seq_arr)
    scores = np.zeros(n, dtype=int)
    
    print(f"Processing {n} sequences. This may take time...")
    
    # 2. Iterate through all sequences and count similars
    # Using a nested loop with explicit similarity check
    for i in range(n):
        if i % 100 == 0: print(f"Scoring sequence {i}/{n}", end='\r')
        count = 0
        target = seq_arr[i]
        
        # Inner loop: compare against every other sequence in split 1
        for j in range(n):
            if i == j: continue # Optional: don't count self, or do (doesn't change ranking)
            if check_similarity(target, seq_arr[j], e):
                count += 1
        
        scores[i] = count

    print("") # clear line
    
    # 3. Sort and Keep top c%
    # Combine sequence and score
    scored_patterns = []
    for i in range(n):
        scored_patterns.append({'sequence': seq_arr[i], 'score': scores[i]})
        
    # Sort descending by frequency score
    scored_patterns.sort(key=lambda x: x['score'], reverse=True)
    
    # Slice top c%
    top_n = int(len(scored_patterns) * c)
    if top_n < 1: top_n = 1
    
    top_patterns = [x['sequence'] for x in scored_patterns[:top_n]]
    
    print(f"Selected {len(top_patterns)} patterns (Top {c*100}% most frequent).")
    return top_patterns

def completesimilarbeginnings(target_df, model_patterns, d, e, raw_df=None):
    results = []
    cols = ['open_pct', 'high_pct', 'low_pct', 'close_pct']
    data = target_df[cols].values
    beg_len = d - 1
    
    for i in range(len(data) - beg_len): 
        current_seq = data[i : i + beg_len]
        prediction = 0 
        match_found = False
        
        # Compare current beginning against model patterns beginnings
        for pattern in model_patterns:
            pattern_beg = pattern[:beg_len]
            
            if check_similarity(current_seq, pattern_beg, e):
                outcome_candle = pattern[-1]
                close_change = outcome_candle[3] # close_pct
                
                if close_change > 0: prediction = 1
                elif close_change < 0: prediction = -1
                match_found = True
                break 
        
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

# --- Threading & Live Logic ---

def live_loop_thread():
    global current_prediction, live_log
    print("Live prediction loop started.")
    
    while True:
        try:
            now = datetime.utcnow()
            tf_seconds = 0
            if 'm' in TIMEFRAME: tf_seconds = int(TIMEFRAME.replace('m','')) * 60
            elif 'h' in TIMEFRAME: tf_seconds = int(TIMEFRAME.replace('h','')) * 3600
            elif 'd' in TIMEFRAME: tf_seconds = int(TIMEFRAME.replace('d','')) * 86400
            
            timestamp = now.timestamp()
            next_ts = (int(timestamp / tf_seconds) + 1) * tf_seconds
            wait_seconds = next_ts - timestamp + 5 
            if wait_seconds < 0: wait_seconds = 0
            
            time.sleep(wait_seconds)
            
            # Fetch & Predict
            raw_df = fetch(TIMEFRAME, SYMBOL, None, None)
            derived_df = deriveround(raw_df)
            
            # Input sequence is from the candles BEFORE the current open one
            # raw_df[-1] is current open. raw_df[-2] is last closed.
            # Sequence ends at -2. Start is -2 - (D_LEN-1) + 1 = -D_LEN
            input_seq_df = derived_df.iloc[-(D_LEN): -1] 
            
            cols = ['open_pct', 'high_pct', 'low_pct', 'close_pct']
            seq_values = input_seq_df[cols].values
            
            pred_dir = 0
            for pattern in model_patterns:
                pattern_beg = pattern[:D_LEN-1]
                if check_similarity(seq_values, pattern_beg, E_SIM):
                    outcome_candle = pattern[-1]
                    if outcome_candle[3] > 0: pred_dir = 1
                    elif outcome_candle[3] < 0: pred_dir = -1
                    break
            
            entry_price = raw_df.iloc[-1]['open']
            
            pred_obj = {
                'timestamp': datetime.utcnow(),
                'input_candles': raw_df.iloc[-(D_LEN): -1][['open','close']].values.tolist(),
                'prediction': "LONG" if pred_dir == 1 else ("SHORT" if pred_dir == -1 else "FLAT"),
                'entry': entry_price,
                'status': 'OPEN',
                'exit': None,
                'pnl': 0
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
    img = io.BytesIO()
    plt.figure(figsize=(12, 5))
    
    dates = [r['timestamp'] for r in backtest_results]
    pnls = np.cumsum([r['pnl'] for r in backtest_results])
    
    plt.subplot(1, 2, 1)
    if len(dates) > 0: plt.plot(dates, pnls, label='Backtest PnL')
    plt.title("Backtest PnL")
    plt.xticks(rotation=45)
    
    if recent_perf_results:
        dates_rec = [r['timestamp'] for r in recent_perf_results]
        pnls_rec = np.cumsum([r['pnl'] for r in recent_perf_results])
        plt.subplot(1, 2, 2)
        plt.plot(dates_rec, pnls_rec, color='orange', label='14d Performance')
        plt.title("Recent 14d Performance")
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    acc, total, _ = get_accuracy_metrics(backtest_results)
    
    html = f"""
    <html>
    <head><title>Trading Bot</title>
    <style>
        body {{ font-family: monospace; margin: 20px; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; font-size: 0.9em; }}
        th, td {{ border: 1px solid #ddd; padding: 5px; text-align: left; }}
        th {{ background-color: #eee; }}
        .win {{ color: green; font-weight: bold; }}
        .loss {{ color: red; font-weight: bold; }}
    </style>
    </head>
    <body>
        <h1>Bot Status: {SYMBOL} {TIMEFRAME}</h1>
        <div>
            <b>Backtest Accuracy:</b> {acc:.2%} ({total} trades)<br>
            <b>Current Prediction:</b> {current_prediction.get('prediction','N/A')} @ {current_prediction.get('entry',0)}
        </div>
        <br>
        <img src="data:image/png;base64,{plot_url}" style="width:100%; max-width:1000px">
        
        <h3>Live Log</h3>
        <table>
            <tr><th>Time</th><th>Pred</th><th>Entry</th><th>Exit</th><th>Outcome</th><th>PnL</th></tr>
            {''.join([f"<tr><td>{r['timestamp']}</td><td>{r['prediction']}</td><td>{r['entry']}</td><td>{r['exit']}</td><td class='{r.get('outcome','').lower()}'>{r.get('outcome','OPEN')}</td><td>{r['pnl']:.4f}</td></tr>" for r in reversed(live_log)])}
        </table>
        
        <h3>Recent Performance (14d)</h3>
        <table>
            <tr><th>Time</th><th>Pred</th><th>Entry</th><th>Exit</th><th>Outcome</th><th>PnL</th></tr>
             {''.join([f"<tr><td>{r['timestamp']}</td><td>{r['prediction']}</td><td>{r['entry']}</td><td>{r['exit']}</td><td class='{r['outcome'].lower()}'>{r['outcome']}</td><td>{r['pnl']:.4f}</td></tr>" for r in reversed(recent_perf_results)])}
        </table>
        
        <h3>Backtest History</h3>
        <div style="height:400px;overflow-y:scroll">
        <table>
            <tr><th>Time</th><th>Pred</th><th>Entry</th><th>Exit</th><th>Outcome</th><th>PnL</th></tr>
             {''.join([f"<tr><td>{r['timestamp']}</td><td>{r['prediction']}</td><td>{r['entry']}</td><td>{r['exit']}</td><td class='{r['outcome'].lower()}'>{r['outcome']}</td><td>{r['pnl']:.4f}</td></tr>" for r in reversed(backtest_results)])}
        </table>
        </div>
    </body>
    </html>
    """
    return render_template_string(html)

@app.route('/api')
def api_endpoint():
    return jsonify({
        'current_prediction': current_prediction,
        'live_log': live_log,
        'recent_performance': recent_perf_results,
        'backtest_summary': {'count': len(backtest_results), 'data': backtest_results}
    })

def main():
    global model_patterns, backtest_results, recent_perf_results
    
    df = fetch(TIMEFRAME, SYMBOL, START_STR, END_STR)
    df_derived = deriveround(df)
    
    train_df, test_df = split(df_derived, B_SPLIT)
    train_raw, test_raw = split(df, B_SPLIT)
    
    model_patterns = gettop(train_df, C_TOP, D_LEN, E_SIM)
    
    print("Running Backtest...")
    backtest_results = completesimilarbeginnings(test_df, model_patterns, D_LEN, E_SIM, test_raw)
    
    print("Calculating Recent Performance...")
    two_weeks_ago = datetime.utcnow() - timedelta(days=14)
    mask = df['datetime'] > two_weeks_ago
    recent_perf_results = completesimilarbeginnings(df_derived[mask], model_patterns, D_LEN, E_SIM, df[mask])
    
    acc, _, _ = get_accuracy_metrics(backtest_results)
    print(f"Backtest Accuracy: {acc:.2%}")
    
    t = threading.Thread(target=live_loop_thread)
    t.daemon = True
    t.start()
    
    print(f"Serving on port {API_PORT}")
    app.run(host='0.0.0.0', port=API_PORT)

if __name__ == '__main__':
    main()
