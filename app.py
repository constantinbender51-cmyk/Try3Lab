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
from datetime import datetime
from collections import Counter, defaultdict
import json

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

# ---------------------------------------------------------
# 1. Fetch 1h ohlc from binance
# ---------------------------------------------------------
def fetch_binance_data(symbol, interval, start_str, end_str):
    print(f"Fetching {symbol} {interval} data from {start_str} to {end_str}...")
    base_url = "https://api.binance.com/api/v3/klines"
    
    start_ts = int(pd.Timestamp(start_str).timestamp() * 1000)
    end_ts = int(pd.Timestamp(end_str).timestamp() * 1000)
    
    all_data = []
    
    while start_ts < end_ts:
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': start_ts,
            'endTime': end_ts,
            'limit': 1000
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
# Processing & Backtesting Logic (Refactored for Loop)
# ---------------------------------------------------------
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
    
    if verbose:
        print(f"Grid: {grid_percent*100}% | Size: {grid_size} | Precision: {needed_precision}")

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
    
    # Train
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
        
        # Calculate Percentage Return for Sharpe (PnL / Investment)
        # Investment is effectively the current price (assuming 1 unit)
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
    
    results_summary = []
    
    for gp in GRID_SEARCH_VALUES:
        res = evaluate_strategy(df, gp, verbose=False)
        
        print(f"Grid: {gp*100:5.2f}% | Sharpe: {res['sharpe']:6.3f} | Acc: {res['accuracy']:5.2f}% | PnL: {res['cumulative_pnl']:8.4f}")
        
        results_summary.append((gp, res['sharpe']))
        
        if res['sharpe'] > best_sharpe:
            best_sharpe = res['sharpe']
            best_result = res
            
    print("-" * 60)
    print(f"Best Grid Percent: {best_result['grid_percent']*100}% with Sharpe: {best_result['sharpe']:.4f}")
    return best_result

# ---------------------------------------------------------
# 6. Serve plot and table
# ---------------------------------------------------------
def create_plot(df, test_results, accuracy, total_pnl, symbol, grid_percent):
    plt.figure(figsize=(12, 6))
    plt.plot(df['open_time'], df['close'], label='Price', color='gray', alpha=0.3)
    
    # Filter df to match test_results timeframe for alignment if needed, 
    # but strictly we just plot the PnL curve overlay
    
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

# Global placeholders for the server
global_best_result = None
global_plot_b64 = None

class BacktestHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        
        # Unpack global results
        res = global_best_result
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
            plot_data=global_plot_b64,
            json_data=json_str,
            precision=needed_precision
        )
        
        self.wfile.write(html_content.encode('utf-8'))

# ---------------------------------------------------------
# Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    df_main = fetch_binance_data(SYMBOL, INTERVAL, START_TIME, END_TIME)
    
    if df_main.empty:
        print("No data fetched. Exiting.")
        sys.exit(1)
        
    # Run Grid Search
    best_result = run_grid_search(df_main)
    
    # Store in global variables for the request handler
    global_best_result = best_result
    
    print("Generating plot for best result...")
    global_plot_b64 = create_plot(
        df_main, 
        best_result['test_results'], 
        best_result['accuracy'], 
        best_result['cumulative_pnl'],
        SYMBOL,
        best_result['grid_percent']
    )
    
    print(f"Starting server on port {PORT}...")
    print(f"Open your browser at http://localhost:{PORT}")
    
    with socketserver.TCPServer(("", PORT), BacktestHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")
            httpd.server_close()
