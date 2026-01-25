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
# 7. All parameters defined at the top
# ---------------------------------------------------------
SYMBOL = 'SHIBUSDT'   # Changed to SHIB to demonstrate robustness
INTERVAL = '1h'
START_TIME = '2024-01-01' # Adjusted for SHIB data availability/relevance
END_TIME = '2026-01-01'
PORT = 8080
GRID_PERCENT = 0.01  # 1%
TRAIN_SPLIT_RATIO = 0.7

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
# Processing & Backtesting Logic
# ---------------------------------------------------------
def run_backtest(df):
    # 2. Round close to 1% * first close
    first_close = df['close'].iloc[0]
    grid_size = first_close * GRID_PERCENT
    
    # --- DYNAMIC PRECISION LOGIC ---
    # Log10 gives us the magnitude (e.g., 0.01 -> -2, 0.0001 -> -4)
    # We take the ceiling of the negative log to get decimal places needed.
    # We add 2 extra digits of safety to prevent float artifacts.
    if grid_size == 0:
        needed_precision = 8
    else:
        needed_precision = int(math.ceil(-math.log10(grid_size))) + 2
    
    # Cap precision to avoid excessive length (Binance max is usually 8)
    needed_precision = max(2, min(needed_precision, 10))
    
    print(f"First Close: {first_close}")
    print(f"Grid Size (1%): {grid_size:.{needed_precision}f}")
    print(f"Dynamic Precision set to: {needed_precision} decimals")

    # Rounding logic: round(value / step) * step
    # We round the FINAL result to 'needed_precision' to clean artifacts
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
    
    # 3. Split train/test
    split_idx = int(len(data) * TRAIN_SPLIT_RATIO)
    train_df = data.iloc[:split_idx]
    test_df = data.iloc[split_idx:]
    
    print(f"Training samples: {len(train_df)}, Test samples: {len(test_df)}")
    
    # 4. Train
    sequence_map = defaultdict(list)
    for _, row in train_df.iterrows():
        seq = (row['t_2'], row['t_1'], row['t_0'])
        sequence_map[seq].append(row['target_direction'])
        
    model = {}
    for seq, directions in sequence_map.items():
        counts = Counter(directions)
        most_common = counts.most_common(1)[0][0]
        model[seq] = most_common
        
    # 5. Test
    results = []
    correct_predictions = 0
    total_predictions = 0
    cumulative_pnl = 0.0
    
    test_results_list = []
    
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
    
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Total PnL: {cumulative_pnl:.8f}")
    
    return train_df, test_df, test_results_list, accuracy, cumulative_pnl, grid_size, needed_precision

# ---------------------------------------------------------
# 6. Serve plot and table
# ---------------------------------------------------------
def create_plot(df, test_results, accuracy, total_pnl):
    plt.figure(figsize=(12, 6))
    plt.plot(df['open_time'], df['close'], label='Price', color='gray', alpha=0.3)
    
    test_times = [x['time_t'] for x in test_results]
    test_pnl = [x['cum_pnl'] for x in test_results]
    
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax2.plot(test_times, test_pnl, label='Strategy PnL', color='blue', linewidth=1.5)
    
    ax1.set_ylabel('Price (USDT)')
    ax2.set_ylabel('Cumulative PnL (USDT)')
    plt.title(f'Backtest: {SYMBOL} | Acc: {accuracy:.2f}% | PnL: {total_pnl:.6f}')
    
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
    </style>
</head>
<body>
    <div class="container">
        <h2>Backtest Report: {symbol}</h2>
        
        <div class="stats">
            <strong>Interval:</strong> {interval} | 
            <strong>Start:</strong> {start} | 
            <strong>End:</strong> {end} |
            <strong>Grid Size:</strong> {grid_size} <br>
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
        const precision = {precision}; // Passed from Python

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
                
                // Use the calculated precision for display
                const rawP = precision + 1; // Show slightly more for raw
                
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
        
        # Format grid size string based on precision
        grid_fmt = f"{grid_size:.{needed_precision}f}"
        pnl_fmt = f"{total_pnl:.{needed_precision}f}"
        
        html_content = HTML_TEMPLATE.format(
            symbol=SYMBOL,
            interval=INTERVAL,
            start=START_TIME,
            end=END_TIME,
            grid_size=grid_fmt,
            accuracy=accuracy,
            pnl=pnl_fmt,
            plot_data=plot_b64,
            json_data=json_str,
            precision=needed_precision
        )
        
        self.wfile.write(html_content.encode('utf-8'))

# ---------------------------------------------------------
# Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    df = fetch_binance_data(SYMBOL, INTERVAL, START_TIME, END_TIME)
    
    if df.empty:
        print("No data fetched. Exiting.")
        sys.exit(1)
        
    train_df, test_df, test_results, accuracy, total_pnl, grid_size, needed_precision = run_backtest(df)
    
    print("Generating plot...")
    plot_b64 = create_plot(df, test_results, accuracy, total_pnl)
    
    print(f"Starting server on port {PORT}...")
    print(f"Open your browser at http://localhost:{PORT}")
    
    with socketserver.TCPServer(("", PORT), BacktestHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")
            httpd.server_close()
