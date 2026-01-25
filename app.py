import http.server
import socketserver
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import io
import base64
import time
import sys
import math
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import json
import threading
import os

# ---------------------------------------------------------
# 7. Parameters
# ---------------------------------------------------------
SYMBOL = 'SHIBUSDT'
INTERVAL = '1h'
START_TIME = '2024-01-01'
END_TIME = '2026-01-01'
PORT = 8080
TRAIN_SPLIT_RATIO = 0.7

# Grid Search Parameters
# Search from 0.1% to 5% grid sizes
GRID_SEARCH_VALUES = [0.001, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]

# Global Storage for Server and Live Loop
GLOBAL_BEST_RESULT = None
GLOBAL_PLOT_B64 = None
GLOBAL_MODEL = {} # Stores the trained pattern map for live use
GLOBAL_GRID_SIZE = 0.0
GLOBAL_PRECISION = 8

# ---------------------------------------------------------
# 1. Fetch Data
# ---------------------------------------------------------
def fetch_binance_data(symbol, interval, start_str, end_str=None, limit=1000):
    """
    Fetches historical data. If end_str is None, fetches most recent data suitable for live loop.
    """
    base_url = "https://api.binance.com/api/v3/klines"
    
    # If end_str is provided, we are in backtest mode (historical range)
    if end_str:
        print(f"Fetching {symbol} {interval} data from {start_str} to {end_str}...")
        start_ts = int(pd.Timestamp(start_str).timestamp() * 1000)
        end_ts = int(pd.Timestamp(end_str).timestamp() * 1000)
        
        all_data = []
        
        while start_ts < end_ts:
            params = {
                'symbol': symbol,
                'interval': interval,
                'startTime': start_ts,
                'endTime': end_ts,
                'limit': limit
            }
            try:
                response = requests.get(base_url, params=params)
                data = response.json()
                if not isinstance(data, list) or len(data) == 0:
                    break
                all_data.extend(data)
                last_close_time = data[-1][6]
                start_ts = last_close_time + 1
                time.sleep(0.05) 
                sys.stdout.write(f"\rFetched {len(all_data)} candles...")
                sys.stdout.flush()
            except Exception as e:
                print(f"Error fetching data: {e}")
                break
        print("\nData fetch complete.")
        
    else:
        # Live mode: just fetch recent candles
        # We need at least 3 completed candles for the sequence (T-2, T-1, T0)
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': 5 
        }
        try:
            response = requests.get(base_url, params=params)
            all_data = response.json()
        except Exception as e:
            print(f"Error fetching live data: {e}")
            return pd.DataFrame()

    df = pd.DataFrame(all_data, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'trades', 
        'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])
    
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df['close'] = df['close'].astype(float)
    
    return df[['open_time', 'close']]

# ---------------------------------------------------------
# Helper: Sharpe Calculation
# ---------------------------------------------------------
def calculate_sharpe_ratio(returns_series, periods_per_year=24*365):
    """
    Calculates Annualized Sharpe Ratio.
    Assumes Risk Free Rate = 0 for simplicity in crypto context.
    """
    if len(returns_series) < 2:
        return 0.0
    mean_ret = np.mean(returns_series)
    std_ret = np.std(returns_series)
    if std_ret == 0:
        return 0.0
    # Annualize
    sharpe = (mean_ret / std_ret) * np.sqrt(periods_per_year)
    return sharpe

# ---------------------------------------------------------
# Processing & Backtesting Logic
# ---------------------------------------------------------
def train_model(df, grid_size, needed_precision):
    """
    Re-trains the model on the provided dataframe to generate the lookup dictionary.
    Used for final model generation after grid search.
    """
    # Rounding logic
    df = df.copy()
    df['rounded_close'] = ((df['close'] / grid_size).round() * grid_size).round(needed_precision)
    
    # Targets
    df['next_rounded'] = df['rounded_close'].shift(-1)
    
    conditions = [
        df['next_rounded'] > df['rounded_close'],
        df['next_rounded'] < df['rounded_close']
    ]
    choices = ['UP', 'DOWN']
    df['target_direction'] = np.select(conditions, choices, default='FLAT')
    
    # Sequences
    df['t_0'] = df['rounded_close']
    df['t_1'] = df['rounded_close'].shift(1)
    df['t_2'] = df['rounded_close'].shift(2)
    
    data = df.dropna().copy()
    
    # Train Map
    sequence_map = defaultdict(list)
    for _, row in data.iterrows():
        seq = (row['t_2'], row['t_1'], row['t_0'])
        sequence_map[seq].append(row['target_direction'])
        
    final_model = {}
    for seq, directions in sequence_map.items():
        counts = Counter(directions)
        most_common = counts.most_common(1)[0][0]
        final_model[seq] = most_common
        
    return final_model

def evaluate_strategy(df_original, grid_percent, verbose=False):
    """
    Runs the strategy for a specific grid_percent.
    Returns metrics needed for optimization and reporting.
    """
    # Work on a copy to avoid SettingWithCopy warnings on the main df
    df = df_original.copy()
    
    first_close = df['close'].iloc[0]
    grid_size = first_close * grid_percent
    
    # --- DYNAMIC PRECISION LOGIC ---
    if grid_size == 0:
        needed_precision = 8
    else:
        needed_precision = int(math.ceil(-math.log10(grid_size))) + 2
    needed_precision = max(2, min(needed_precision, 10))
    
    # Rounding logic
    df['rounded_close'] = ((df['close'] / grid_size).round() * grid_size).round(needed_precision)
    
    # Prepare Targets
    df['next_rounded'] = df['rounded_close'].shift(-1)
    df['next_close_raw'] = df['close'].shift(-1)
    
    conditions = [
        df['next_rounded'] > df['rounded_close'],
        df['next_rounded'] < df['rounded_close']
    ]
    choices = ['UP', 'DOWN']
    df['target_direction'] = np.select(conditions, choices, default='FLAT')
    
    # Create sequences
    df['t_0'] = df['rounded_close']
    df['t_1'] = df['rounded_close'].shift(1)
    df['t_2'] = df['rounded_close'].shift(2)

    df['raw_t_0'] = df['close']
    df['raw_t_1'] = df['close'].shift(1)
    df['raw_t_2'] = df['close'].shift(2)
    
    data = df.dropna().copy()
    
    # Split train/test
    split_idx = int(len(data) * TRAIN_SPLIT_RATIO)
    train_df = data.iloc[:split_idx]
    test_df = data.iloc[split_idx:]
    
    # Train (Inline for speed)
    sequence_map = defaultdict(list)
    for _, row in train_df.iterrows():
        seq = (row['t_2'], row['t_1'], row['t_0'])
        sequence_map[seq].append(row['target_direction'])
        
    model = {}
    for seq, directions in sequence_map.items():
        counts = Counter(directions)
        most_common = counts.most_common(1)[0][0]
        model[seq] = most_common
        
    # Test
    correct_predictions = 0
    total_predictions = 0
    cumulative_pnl = 0.0
    
    test_results_list = []
    hourly_returns = []
    
    for idx, row in test_df.iterrows():
        seq = (row['t_2'], row['t_1'], row['t_0'])
        
        prediction = model.get(seq, 'FLAT') 
        actual = row['target_direction']
        
        if prediction == actual:
            correct_predictions += 1
        total_predictions += 1
        
        curr_price = row['close']
        next_price = row['next_close_raw']
        
        trade_pnl = 0.0
        
        # Calculate PnL
        if prediction == 'UP':
            trade_pnl = next_price - curr_price
        elif prediction == 'DOWN':
            trade_pnl = curr_price - next_price
            
        cumulative_pnl += trade_pnl
        
        pct_return = trade_pnl / curr_price if curr_price > 0 else 0
        hourly_returns.append(pct_return)
        
        test_results_list.append({
            'time_t': row['open_time'],
            'rnd_t_2': row['t_2'],
            'rnd_t_1': row['t_1'],
            'rnd_t_0': row['t_0'],
            'raw_t_2': row['raw_t_2'],
            'raw_t_1': row['raw_t_1'],
            'raw_t_0': row['raw_t_0'],
            'prediction': prediction,
            'actual': actual,
            'pnl': trade_pnl,
            'cum_pnl': cumulative_pnl,
            'next_price_raw': next_price
        })
        
    accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
    sharpe = calculate_sharpe_ratio(hourly_returns)
    
    return {
        'sharpe': sharpe,
        'accuracy': accuracy,
        'cumulative_pnl': cumulative_pnl,
        'grid_size': grid_size,
        'needed_precision': needed_precision,
        'test_results': test_results_list,
        'grid_percent': grid_percent
    }

# ---------------------------------------------------------
# Grid Search Driver
# ---------------------------------------------------------
def run_grid_search(df):
    print(f"\n--- Starting Grid Search for Sharpe Optimization ---")
    print(f"Values to test: {GRID_SEARCH_VALUES}")
    
    best_sharpe = -float('inf')
    best_result = None
    
    for gp in GRID_SEARCH_VALUES:
        res = evaluate_strategy(df, gp, verbose=False)
        print(f"Grid: {gp*100:5.2f}% | Sharpe: {res['sharpe']:6.3f} | Acc: {res['accuracy']:5.2f}% | PnL: {res['cumulative_pnl']:8.4f}")
        
        if res['sharpe'] > best_sharpe:
            best_sharpe = res['sharpe']
            best_result = res
            
    print("-" * 60)
    print(f"Best Grid Percent: {best_result['grid_percent']*100}% with Sharpe: {best_result['sharpe']:.4f}")
    return best_result

# ---------------------------------------------------------
# Visualization & Server
# ---------------------------------------------------------
def create_plot(df, test_results, accuracy, total_pnl, symbol, grid_percent):
    plt.figure(figsize=(12, 6))
    plt.plot(df['open_time'], df['close'], label='Price', color='gray', alpha=0.3)
    
    test_times = [x['time_t'] for x in test_results]
    test_pnl = [x['cum_pnl'] for x in test_results]
    
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax2.plot(test_times, test_pnl, label='Strategy PnL', color='blue', linewidth=1.5)
    
    ax1.set_ylabel('Price (USDT)')
    ax2.set_ylabel('Cumulative PnL (USDT)')
    plt.title(f'Optimized Backtest: {symbol} (Grid={grid_percent*100}%) | Acc: {accuracy:.2f}% | PnL: {total_pnl:.6f}')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return image_base64

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Binance Backtest Results</title>
    <style>
        body {{ font-family: monospace; margin: 20px; color: #333; }}
        .container {{ max-width: 95%; margin: 0 auto; }}
        img {{ max-width: 100%; height: auto; border: 1px solid #ccc; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; font-size: 12px; }}
        th, td {{ padding: 6px 10px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #eee; }}
        .pagination {{ margin-top: 20px; text-align: center; }}
        button {{ padding: 5px 15px; margin: 0 5px; cursor: pointer; }}
        .up {{ color: green; font-weight: bold; }}
        .down {{ color: red; font-weight: bold; }}
        .stats {{ background: #f4f4f4; padding: 15px; border-radius: 4px; margin-bottom: 20px; }}
        .raw-price {{ color: #666; font-style: italic; }}
        .highlight {{ color: #007bff; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="container">
        <h2>Optimized Backtest Report: {symbol}</h2>
        
        <div class="stats">
            <strong>Interval:</strong> {interval} | 
            <strong>Start:</strong> {start} | 
            <strong>End:</strong> {end} <br>
            <strong>Optimized Grid:</strong> <span class="highlight">{grid_percent:.2f}%</span> (Size: {grid_size}) <br>
            <strong>Sharpe Ratio:</strong> <span class="highlight">{sharpe:.4f}</span> |
            <strong>Accuracy:</strong> {accuracy:.2f}% | 
            <strong>Total PnL:</strong> {pnl} USDT
        </div>

        <img src="data:image/png;base64,{plot_data}" />

        <h3>Prediction Log</h3>
        <div class="pagination">
            <button onclick="prevPage()">Previous</button>
            <span id="pageInfoTop"></span>
            <button onclick="nextPage()">Next</button>
        </div>
        
        <table id="resultsTable">
            <thead>
                <tr>
                    <th>Time (T)</th>
                    <th>Raw Input [T-2, T-1, T]</th>
                    <th>Grid Input [T-2, T-1, T]</th>
                    <th>Prediction</th>
                    <th>Target Raw (T+1)</th>
                    <th>Actual Dir</th>
                    <th>PnL</th>
                </tr>
            </thead>
            <tbody id="tableBody"></tbody>
        </table>
    </div>

    <script>
        const data = {json_data};
        const rowsPerPage = 50;
        let currentPage = 1;
        const precision = {precision}; 

        function renderTable() {{
            const tbody = document.getElementById('tableBody');
            tbody.innerHTML = '';

            const start = (currentPage - 1) * rowsPerPage;
            const end = start + rowsPerPage;
            const pageData = data.slice(start, end);

            pageData.forEach(row => {{
                const tr = document.createElement('tr');
                
                const predClass = row.prediction === 'UP' ? 'up' : (row.prediction === 'DOWN' ? 'down' : '');
                const actualClass = row.actual === 'UP' ? 'up' : (row.actual === 'DOWN' ? 'down' : '');
                const pnlColor = row.pnl >= 0 ? 'green' : 'red';
                
                const rawP = precision + 2; 
                
                tr.innerHTML = `
                    <td>${{row.time_t}}</td>
                    <td class="raw-price">[${{row.raw_t_2.toFixed(rawP)}}, ${{row.raw_t_1.toFixed(rawP)}}, ${{row.raw_t_0.toFixed(rawP)}}]</td>
                    <td>[${{row.rnd_t_2.toFixed(precision)}}, ${{row.rnd_t_1.toFixed(precision)}}, ${{row.rnd_t_0.toFixed(precision)}}]</td>
                    <td class="${{predClass}}">${{row.prediction}}</td>
                    <td>${{row.next_price_raw.toFixed(rawP)}}</td>
                    <td class="${{actualClass}}">${{row.actual}}</td>
                    <td style="color: ${{pnlColor}}">${{row.pnl.toFixed(precision)}}</td>
                `;
                tbody.appendChild(tr);
            }});

            const info = `Page ${{currentPage}} of ${{Math.ceil(data.length / rowsPerPage)}}`;
            document.getElementById('pageInfoTop').innerText = info;
        }}

        function prevPage() {{
            if (currentPage > 1) {{
                currentPage--;
                renderTable();
            }}
        }}

        function nextPage() {{
            if (currentPage * rowsPerPage < data.length) {{
                currentPage++;
                renderTable();
            }}
        }}

        renderTable();
    </script>
</body>
</html>
"""

class BacktestHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        
        # Unpack global results
        res = GLOBAL_BEST_RESULT
        if not res:
            self.wfile.write(b"Results not ready yet. Please wait for backtest to finish.")
            return

        test_results = res['test_results']
        needed_precision = res['needed_precision']
        
        js_data = []
        for r in test_results:
            js_data.append({
                'time_t': str(r['time_t']),
                'raw_t_2': r['raw_t_2'],
                'raw_t_1': r['raw_t_1'],
                'raw_t_0': r['raw_t_0'],
                'rnd_t_2': r['rnd_t_2'],
                'rnd_t_1': r['rnd_t_1'],
                'rnd_t_0': r['rnd_t_0'],
                'prediction': r['prediction'],
                'next_price_raw': r['next_price_raw'],
                'actual': r['actual'],
                'pnl': r['pnl']
            })
            
        json_str = json.dumps(js_data)
        
        grid_fmt = f"{res['grid_size']:.{needed_precision}f}"
        pnl_fmt = f"{res['cumulative_pnl']:.{needed_precision}f}"
        
        html_content = HTML_TEMPLATE.format(
            symbol=SYMBOL,
            interval=INTERVAL,
            start=START_TIME,
            end=END_TIME,
            grid_size=grid_fmt,
            grid_percent=res['grid_percent']*100,
            sharpe=res['sharpe'],
            accuracy=res['accuracy'],
            pnl=pnl_fmt,
            plot_data=GLOBAL_PLOT_B64,
            json_data=json_str,
            precision=needed_precision
        )
        
        self.wfile.write(html_content.encode('utf-8'))

# ---------------------------------------------------------
# Live Prediction Logic
# ---------------------------------------------------------
def save_live_prediction(timestamp, close_price, sequence, prediction):
    """
    Appends the prediction to a CSV log file.
    """
    file_exists = os.path.isfile("live_prediction_log.csv")
    
    with open("live_prediction_log.csv", "a") as f:
        if not file_exists:
            f.write("timestamp,close_price,sequence_t2,sequence_t1,sequence_t0,prediction\n")
        
        seq_str = f"{sequence[0]},{sequence[1]},{sequence[2]}"
        f.write(f"{timestamp},{close_price},{seq_str},{prediction}\n")
    
    print(f"Logged prediction: {prediction} for seq {sequence}")

def live_loop():
    print(f"\n--- Starting Live Prediction Loop for {SYMBOL} ---")
    
    # Ensure we have the global model and grid size from the backtest
    if not GLOBAL_MODEL or GLOBAL_GRID_SIZE == 0:
        print("Error: Global model not trained. Live loop aborting.")
        return

    while True:
        # 1. Calculate time until next hour + 5 seconds buffer
        now = datetime.now()
        next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
        
        # Add buffer to ensure exchange data is ready
        fetch_time = next_hour + timedelta(seconds=5)
        
        sleep_seconds = (fetch_time - now).total_seconds()
        
        print(f"\n[Live] Current time: {now.strftime('%H:%M:%S')}")
        print(f"[Live] Sleeping for {sleep_seconds:.0f} seconds until {fetch_time.strftime('%H:%M:%S')}...")
        
        time.sleep(sleep_seconds)
        
        # 2. Wake up and fetch recent data
        print(f"\n[Live] Waking up. Fetching recent candles for {SYMBOL}...")
        df_live = fetch_binance_data(SYMBOL, INTERVAL, start_str=None, end_str=None)
        
        if len(df_live) < 3:
            print("[Live] Error: Not enough data fetched to form sequence.")
            continue
            
        # We need the 3 most recently COMPLETED candles.
        # Binance returns the last candle (index -1) as the currently OPEN candle.
        # So completed candles are at indices -4, -3, -2 relative to list end (or just slice appropriately).
        
        # Get last 3 completed closes. 
        # Index -2 is the one that just closed 5 seconds ago.
        # Index -3 is T-1
        # Index -4 is T-2
        
        try:
            # Taking the slice 0:-1 drops the currently open candle
            completed_df = df_live.iloc[:-1].tail(3).copy()
            
            if len(completed_df) < 3:
                print("[Live] Error: Insufficient completed history.")
                continue

            closes = completed_df['close'].values
            
            # 3. Rounding
            rounded_closes = ((closes / GLOBAL_GRID_SIZE).round() * GLOBAL_GRID_SIZE).round(GLOBAL_PRECISION)
            
            t_2 = rounded_closes[0]
            t_1 = rounded_closes[1]
            t_0 = rounded_closes[2] # The candle that just closed
            
            seq = (t_2, t_1, t_0)
            
            # 4. Predict
            prediction = GLOBAL_MODEL.get(seq, 'FLAT') # Default to FLAT if unknown seq
            
            # 5. Log
            last_close_time = completed_df['open_time'].iloc[-1] # This is the open time of the candle that just closed
            save_live_prediction(last_close_time, closes[-1], seq, prediction)
            
            print(f"[Live] Candle Closed: {closes[-1]}")
            print(f"[Live] Sequence: {seq}")
            print(f"[Live] Prediction for next candle: {prediction}")
            
        except Exception as e:
            print(f"[Live] Error during processing: {e}")

# ---------------------------------------------------------
# Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    # 1. Backtest & Optimization
    df_main = fetch_binance_data(SYMBOL, INTERVAL, START_TIME, END_TIME)
    
    if df_main.empty:
        print("No data fetched. Exiting.")
        sys.exit(1)
        
    best_result = run_grid_search(df_main)
    GLOBAL_BEST_RESULT = best_result
    GLOBAL_GRID_SIZE = best_result['grid_size']
    GLOBAL_PRECISION = best_result['needed_precision']
    
    # 2. Re-train full model for live use
    # We use the whole dataset (no train/test split) to have maximum knowledge for live predictions
    print("Training final model for live use...")
    GLOBAL_MODEL = train_model(df_main, GLOBAL_GRID_SIZE, GLOBAL_PRECISION)
    
    # 3. Generate Plot
    print("Generating plot for best result...")
    GLOBAL_PLOT_B64 = create_plot(
        df_main, 
        best_result['test_results'], 
        best_result['accuracy'], 
        best_result['cumulative_pnl'],
        SYMBOL,
        best_result['grid_percent']
    )
    
    # 4. Start Server in a separate thread
    server_thread = threading.Thread(target=lambda: socketserver.TCPServer(("", PORT), BacktestHandler).serve_forever())
    server_thread.daemon = True # Daemon thread exits when main program exits
    server_thread.start()
    
    print(f"Server running at http://localhost:{PORT}")
    print(f"Open your browser at http://localhost:{PORT}")
    
    # 5. Enter Live Loop (Main Thread)
    try:
        live_loop()
    except KeyboardInterrupt:
        print("\nStopping...")
        sys.exit(0)
