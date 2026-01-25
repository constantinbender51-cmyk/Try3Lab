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
import csv

# ---------------------------------------------------------
# 7. Parameters (Environment Variables)
# ---------------------------------------------------------
SYMBOL = os.getenv("SYMBOL", 'SHIBUSDT')
INTERVAL = os.getenv("INTERVAL", '1h')
START_TIME = os.getenv("START_TIME", '2024-01-01')
END_TIME = os.getenv("END_TIME", '2026-01-01')
PORT = int(os.getenv("PORT", 8080))
TRAIN_SPLIT_RATIO = float(os.getenv("TRAIN_SPLIT_RATIO", 0.7))

# Parse Grid Search Values from Env (comma separated) or use default
_grid_env = os.getenv("GRID_SEARCH_VALUES")
if _grid_env:
    GRID_SEARCH_VALUES = [float(x.strip()) for x in _grid_env.split(',')]
else:
    # Default: 0.1% to 5%
    GRID_SEARCH_VALUES = [0.001, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]

# Global Storage
GLOBAL_BEST_RESULT = None
GLOBAL_PLOT_B64 = None
GLOBAL_MODEL = {} 
GLOBAL_GRID_SIZE = 0.0
GLOBAL_PRECISION = 8
GLOBAL_LIVE_LOG = [] # In-memory store for live predictions

# ---------------------------------------------------------
# 1. Fetch Data
# ---------------------------------------------------------
def fetch_binance_data(symbol, interval, start_str, end_str=None, limit=1000):
    base_url = "https://api.binance.com/api/v3/klines"
    
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
        # Live mode
        params = {'symbol': symbol, 'interval': interval, 'limit': 5}
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
    if len(returns_series) < 2:
        return 0.0
    mean_ret = np.mean(returns_series)
    std_ret = np.std(returns_series)
    if std_ret == 0:
        return 0.0
    return (mean_ret / std_ret) * np.sqrt(periods_per_year)

# ---------------------------------------------------------
# Processing & Backtesting Logic
# ---------------------------------------------------------
def train_model(df, grid_size, needed_precision):
    df = df.copy()
    df['rounded_close'] = ((df['close'] / grid_size).round() * grid_size).round(needed_precision)
    df['next_rounded'] = df['rounded_close'].shift(-1)
    
    conditions = [
        df['next_rounded'] > df['rounded_close'],
        df['next_rounded'] < df['rounded_close']
    ]
    choices = ['UP', 'DOWN']
    df['target_direction'] = np.select(conditions, choices, default='FLAT')
    
    df['t_0'] = df['rounded_close']
    df['t_1'] = df['rounded_close'].shift(1)
    df['t_2'] = df['rounded_close'].shift(2)
    
    data = df.dropna().copy()
    
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
    df = df_original.copy()
    first_close = df['close'].iloc[0]
    grid_size = first_close * grid_percent
    
    if grid_size == 0:
        needed_precision = 8
    else:
        needed_precision = int(math.ceil(-math.log10(grid_size))) + 2
    needed_precision = max(2, min(needed_precision, 10))
    
    df['rounded_close'] = ((df['close'] / grid_size).round() * grid_size).round(needed_precision)
    df['next_rounded'] = df['rounded_close'].shift(-1)
    df['next_close_raw'] = df['close'].shift(-1)
    
    conditions = [
        df['next_rounded'] > df['rounded_close'],
        df['next_rounded'] < df['rounded_close']
    ]
    choices = ['UP', 'DOWN']
    df['target_direction'] = np.select(conditions, choices, default='FLAT')
    
    df['t_0'] = df['rounded_close']
    df['t_1'] = df['rounded_close'].shift(1)
    df['t_2'] = df['rounded_close'].shift(2)

    df['raw_t_0'] = df['close']
    df['raw_t_1'] = df['close'].shift(1)
    df['raw_t_2'] = df['close'].shift(2)
    
    data = df.dropna().copy()
    split_idx = int(len(data) * TRAIN_SPLIT_RATIO)
    train_df = data.iloc[:split_idx]
    test_df = data.iloc[split_idx:]
    
    sequence_map = defaultdict(list)
    for _, row in train_df.iterrows():
        seq = (row['t_2'], row['t_1'], row['t_0'])
        sequence_map[seq].append(row['target_direction'])
        
    model = {}
    for seq, directions in sequence_map.items():
        counts = Counter(directions)
        model[seq] = counts.most_common(1)[0][0]
        
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
        if prediction == 'UP':
            trade_pnl = next_price - curr_price
        elif prediction == 'DOWN':
            trade_pnl = curr_price - next_price
            
        cumulative_pnl += trade_pnl
        hourly_returns.append(trade_pnl / curr_price if curr_price > 0 else 0)
        
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
    <title>Binance Strategy Dashboard</title>
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
        .section-header {{ margin-top: 40px; border-bottom: 2px solid #333; padding-bottom: 5px; }}
        .live-tag {{ background: #ff4444; color: white; padding: 2px 6px; border-radius: 3px; font-size: 10px; vertical-align: middle; }}
    </style>
</head>
<body>
    <div class="container">
        <h2>Strategy Dashboard: {symbol}</h2>
        
        <div class="stats">
            <strong>Interval:</strong> {interval} | 
            <strong>Start:</strong> {start} | 
            <strong>End:</strong> {end} <br>
            <strong>Optimized Grid:</strong> <span class="highlight">{grid_percent:.2f}%</span> (Size: {grid_size}) <br>
            <strong>Sharpe Ratio:</strong> <span class="highlight">{sharpe:.4f}</span> |
            <strong>Accuracy:</strong> {accuracy:.2f}% | 
            <strong>Total PnL:</strong> {pnl} USDT
        </div>

        <h3 class="section-header">Live Monitor <span class="live-tag">ACTIVE</span></h3>
        <p><em>Updates every hour. Most recent predictions from the active loop.</em></p>
        <table>
            <thead>
                <tr>
                    <th>Timestamp</th>
                    <th>Close Price</th>
                    <th>Pattern Sequence</th>
                    <th>Predicted Direction</th>
                </tr>
            </thead>
            <tbody>
                {live_rows}
            </tbody>
        </table>

        <h3 class="section-header">Backtest Performance</h3>
        <img src="data:image/png;base64,{plot_data}" />

        <h3>Backtest Log</h3>
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

        function prevPage() {{ if (currentPage > 1) {{ currentPage--; renderTable(); }} }}
        function nextPage() {{ if (currentPage * rowsPerPage < data.length) {{ currentPage++; renderTable(); }} }}
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
        
        res = GLOBAL_BEST_RESULT
        if not res:
            self.wfile.write(b"Results not ready yet. Please wait for backtest to finish.")
            return

        # Prepare Backtest Data
        test_results = res['test_results']
        needed_precision = res['needed_precision']
        js_data = []
        for r in test_results:
            js_data.append({
                'time_t': str(r['time_t']),
                'raw_t_2': r['raw_t_2'], 'raw_t_1': r['raw_t_1'], 'raw_t_0': r['raw_t_0'],
                'rnd_t_2': r['rnd_t_2'], 'rnd_t_1': r['rnd_t_1'], 'rnd_t_0': r['rnd_t_0'],
                'prediction': r['prediction'],
                'next_price_raw': r['next_price_raw'],
                'actual': r['actual'], 'pnl': r['pnl']
            })
        json_str = json.dumps(js_data)
        
        # Prepare Live Log Rows
        live_rows_html = ""
        # Reverse to show newest first
        for entry in reversed(GLOBAL_LIVE_LOG):
            pred_class = "up" if entry['prediction'] == 'UP' else ("down" if entry['prediction'] == 'DOWN' else "")
            live_rows_html += f"""
                <tr>
                    <td>{entry['timestamp']}</td>
                    <td>{entry['close_price']}</td>
                    <td>{entry['sequence']}</td>
                    <td class="{pred_class}">{entry['prediction']}</td>
                </tr>
            """
        
        if not live_rows_html:
            live_rows_html = "<tr><td colspan='4'>No live predictions yet. Next prediction scheduled.</td></tr>"

        html_content = HTML_TEMPLATE.format(
            symbol=SYMBOL,
            interval=INTERVAL,
            start=START_TIME,
            end=END_TIME,
            grid_size=f"{res['grid_size']:.{needed_precision}f}",
            grid_percent=res['grid_percent']*100,
            sharpe=res['sharpe'],
            accuracy=res['accuracy'],
            pnl=f"{res['cumulative_pnl']:.{needed_precision}f}",
            plot_data=GLOBAL_PLOT_B64,
            json_data=json_str,
            precision=needed_precision,
            live_rows=live_rows_html
        )
        self.wfile.write(html_content.encode('utf-8'))

# ---------------------------------------------------------
# Live Prediction Logic
# ---------------------------------------------------------
def save_live_prediction(timestamp, close_price, sequence, prediction):
    """
    Appends the prediction to CSV and Global Memory.
    """
    file_exists = os.path.isfile("live_prediction_log.csv")
    
    # 1. Write to CSV
    with open("live_prediction_log.csv", "a") as f:
        if not file_exists:
            f.write("timestamp,close_price,sequence_t2,sequence_t1,sequence_t0,prediction\n")
        seq_str = f"{sequence[0]}|{sequence[1]}|{sequence[2]}" # Using pipe to simplify reading
        f.write(f"{timestamp},{close_price},{seq_str},{prediction}\n")
    
    # 2. Update In-Memory List
    GLOBAL_LIVE_LOG.append({
        'timestamp': str(timestamp),
        'close_price': close_price,
        'sequence': str(sequence),
        'prediction': prediction
    })
    
    print(f"Logged prediction: {prediction} for seq {sequence}")

def load_existing_logs():
    """Reads existing CSV logs into memory on startup."""
    if os.path.isfile("live_prediction_log.csv"):
        try:
            with open("live_prediction_log.csv", "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Reconstruct readable format
                    try:
                        seq_raw = f"({row['sequence_t2']}, {row['sequence_t1']}, {row['sequence_t0']})"
                    except KeyError:
                        # Handle old format if exists or simplified
                        seq_raw = row.get('sequence_t2', 'n/a')
                        
                    GLOBAL_LIVE_LOG.append({
                        'timestamp': row['timestamp'],
                        'close_price': row['close_price'],
                        'sequence': seq_raw,
                        'prediction': row['prediction']
                    })
            print(f"Loaded {len(GLOBAL_LIVE_LOG)} existing live logs.")
        except Exception as e:
            print(f"Error loading existing logs: {e}")

def live_loop():
    print(f"\n--- Starting Live Prediction Loop for {SYMBOL} ---")
    
    if not GLOBAL_MODEL or GLOBAL_GRID_SIZE == 0:
        print("Error: Global model not trained. Live loop aborting.")
        return

    while True:
        now = datetime.now()
        next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
        fetch_time = next_hour + timedelta(seconds=5)
        sleep_seconds = (fetch_time - now).total_seconds()
        
        print(f"\n[Live] Current time: {now.strftime('%H:%M:%S')}")
        print(f"[Live] Sleeping for {sleep_seconds:.0f} seconds until {fetch_time.strftime('%H:%M:%S')}...")
        
        time.sleep(sleep_seconds)
        
        print(f"\n[Live] Waking up. Fetching recent candles for {SYMBOL}...")
        df_live = fetch_binance_data(SYMBOL, INTERVAL, start_str=None, end_str=None)
        
        if len(df_live) < 3:
            print("[Live] Error: Not enough data fetched to form sequence.")
            continue
            
        try:
            completed_df = df_live.iloc[:-1].tail(3).copy()
            if len(completed_df) < 3: continue

            closes = completed_df['close'].values
            rounded_closes = ((closes / GLOBAL_GRID_SIZE).round() * GLOBAL_GRID_SIZE).round(GLOBAL_PRECISION)
            
            seq = (rounded_closes[0], rounded_closes[1], rounded_closes[2])
            prediction = GLOBAL_MODEL.get(seq, 'FLAT')
            
            last_close_time = completed_df['open_time'].iloc[-1]
            save_live_prediction(last_close_time, closes[-1], seq, prediction)
            
            print(f"[Live] Candle Closed: {closes[-1]} | Pred: {prediction}")
            
        except Exception as e:
            print(f"[Live] Error during processing: {e}")

# ---------------------------------------------------------
# Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    # 0. Load existing logs
    load_existing_logs()

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
    
    # 4. Start Server
    server_thread = threading.Thread(target=lambda: socketserver.TCPServer(("", PORT), BacktestHandler).serve_forever())
    server_thread.daemon = True
    server_thread.start()
    
    print(f"Server running at http://localhost:{PORT}")
    
    # 5. Live Loop
    try:
        live_loop()
    except KeyboardInterrupt:
        print("\nStopping...")
        sys.exit(0)
