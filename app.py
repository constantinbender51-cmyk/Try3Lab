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
from datetime import datetime
from collections import Counter, defaultdict

# ---------------------------------------------------------
# 7. All parameters defined at the top
# ---------------------------------------------------------
SYMBOL = 'BTCUSDT'
INTERVAL = '1h'
START_TIME = '2020-01-01'
END_TIME = '2026-01-01'
PORT = 8080
GRID_PERCENT = 0.01  # 1%
TRAIN_SPLIT_RATIO = 0.7

# ---------------------------------------------------------
# 1. Fetch 1h ohlc from binance (Pagination logic included)
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
            
            # Update start_ts to the last close time + 1ms
            last_close_time = data[-1][6]
            start_ts = last_close_time + 1
            
            # Simple rate limit handling
            time.sleep(0.1)
            
            # Progress indicator
            sys.stdout.write(f"\rFetched {len(all_data)} candles...")
            sys.stdout.flush()
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            break
            
    print("\nData fetch complete.")
    
    # Create DataFrame
    df = pd.DataFrame(all_data, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'trades', 
        'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])
    
    # Convert types
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df['close'] = df['close'].astype(float)
    
    # Keep only necessary columns
    return df[['open_time', 'close']]

# ---------------------------------------------------------
# Processing & Backtesting Logic
# ---------------------------------------------------------
def run_backtest(df):
    # 2. Round close to 1% * first close
    first_close = df['close'].iloc[0]
    grid_size = first_close * GRID_PERCENT
    
    print(f"First Close: {first_close}, Grid Size (1%): {grid_size}")
    
    # Rounding logic: round(value / step) * step
    df['rounded_close'] = (df['close'] / grid_size).round() * grid_size
    
    # Prepare Sequences
    # We need sequences of 3 rounded closes -> Next Direction
    # Direction logic: 
    # If next_rounded > curr_rounded -> UP
    # If next_rounded < curr_rounded -> DOWN
    # Else -> FLAT
    
    # Shift to get next value
    df['next_rounded'] = df['rounded_close'].shift(-1)
    df['next_close_raw'] = df['close'].shift(-1)
    
    conditions = [
        df['next_rounded'] > df['rounded_close'],
        df['next_rounded'] < df['rounded_close']
    ]
    choices = ['UP', 'DOWN']
    df['target_direction'] = np.select(conditions, choices, default='FLAT')
    
    # Create sequences of length 3
    # We need input: [t-2, t-1, t] to predict t+1
    # Create lag columns for the sequence
    df['t_0'] = df['rounded_close']
    df['t_1'] = df['rounded_close'].shift(1)
    df['t_2'] = df['rounded_close'].shift(2)
    
    # Drop NaNs created by shifting
    data = df.dropna().copy()
    
    # 3. Split train/test
    split_idx = int(len(data) * TRAIN_SPLIT_RATIO)
    train_df = data.iloc[:split_idx]
    test_df = data.iloc[split_idx:]
    
    print(f"Training samples: {len(train_df)}, Test samples: {len(test_df)}")
    
    # 4. For each unique sequence of 3 find the most frequent next close direction
    # Map: (val1, val2, val3) -> Direction
    sequence_map = defaultdict(list)
    
    for _, row in train_df.iterrows():
        seq = (row['t_2'], row['t_1'], row['t_0'])
        sequence_map[seq].append(row['target_direction'])
        
    # Consolidate to most frequent
    model = {}
    for seq, directions in sequence_map.items():
        counts = Counter(directions)
        most_common = counts.most_common(1)[0][0]
        model[seq] = most_common
        
    # 5. Test on test sequences to obtain accuracy and pnl
    results = []
    correct_predictions = 0
    total_predictions = 0
    cumulative_pnl = 0.0
    
    # Iterate through test set
    test_results_list = []
    
    for idx, row in test_df.iterrows():
        seq = (row['t_2'], row['t_1'], row['t_0'])
        
        # Prediction
        prediction = model.get(seq, 'FLAT') # Default to FLAT if unseen
        
        actual = row['target_direction']
        is_correct = (prediction == actual)
        
        if is_correct:
            correct_predictions += 1
        total_predictions += 1
        
        # PnL Calculation
        # Strategy: 
        # UP prediction -> Long (Profit = Next Close - Curr Close)
        # DOWN prediction -> Short (Profit = Curr Close - Next Close)
        # FLAT -> No trade
        
        # Note: Using raw close prices for PnL to reflect real money, 
        # though logic is based on rounded grid.
        curr_price = row['close']
        next_price = row['next_close_raw']
        
        trade_pnl = 0.0
        if prediction == 'UP':
            trade_pnl = next_price - curr_price
        elif prediction == 'DOWN':
            trade_pnl = curr_price - next_price
            
        cumulative_pnl += trade_pnl
        
        # Save for table (Paginated display)
        # 9. Show input prices with timestamp and target price w/ timestamp
        # We need to recover timestamps for t-2 and t-1
        # Since we are iterating rows, row['open_time'] is time T.
        # T-1 is T - 1h, T-2 is T - 2h
        
        test_results_list.append({
            'time_t': row['open_time'],
            'price_t': row['close'],
            'price_t_rounded': row['t_0'],
            'price_t_1': row['t_1'],
            'price_t_2': row['t_2'],
            'prediction': prediction,
            'actual': actual,
            'pnl': trade_pnl,
            'cum_pnl': cumulative_pnl,
            'next_price': next_price
        })
        
    accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
    
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Total PnL: {cumulative_pnl:.2f}")
    
    return train_df, test_df, test_results_list, accuracy, cumulative_pnl

# ---------------------------------------------------------
# 6. Serve plot and table on port 8080 (Matplotlib + http.server)
# ---------------------------------------------------------
def create_plot(df, test_results):
    plt.figure(figsize=(12, 6))
    
    # Plot entire price history
    plt.plot(df['open_time'], df['close'], label='Price', color='gray', alpha=0.5)
    
    # Plot Equity Curve for the test period
    test_times = [x['time_t'] for x in test_results]
    test_pnl = [x['cum_pnl'] for x in test_results]
    
    # Create a secondary axis for PnL
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax2.plot(test_times, test_pnl, label='Strategy PnL', color='blue')
    
    ax1.set_ylabel('Price (USDT)')
    ax2.set_ylabel('Cumulative PnL (USDT)')
    plt.title(f'Backtest Results: {SYMBOL} | Acc: {accuracy:.2f}% | PnL: {total_pnl:.2f}')
    
    # Save to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return image_base64

# HTML Template with Client-Side Pagination script
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Binance Backtest Results</title>
    <style>
        body {{ font-family: sans-serif; margin: 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; font-size: 0.9em; }}
        th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f2f2f2; }}
        .pagination {{ margin-top: 20px; text-align: center; }}
        button {{ padding: 5px 10px; margin: 0 5px; cursor: pointer; }}
        .up {{ color: green; font-weight: bold; }}
        .down {{ color: red; font-weight: bold; }}
        .stats {{ background: #f9f9f9; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Backtest Report: {symbol}</h1>
        
        <div class="stats">
            <strong>Interval:</strong> {interval} | 
            <strong>Start:</strong> {start} | 
            <strong>End:</strong> {end} <br>
            <strong>Accuracy:</strong> {accuracy:.2f}% | 
            <strong>Total PnL:</strong> {pnl:.2f} USDT
        </div>

        <h3>Equity Curve & Price</h3>
        <img src="data:image/png;base64,{plot_data}" />

        <h3>Prediction Log</h3>
        <div id="table-container">
            <table id="resultsTable">
                <thead>
                    <tr>
                        <th>Time (T)</th>
                        <th>Input Seq (Rounded) [T-2, T-1, T]</th>
                        <th>Actual Price (T)</th>
                        <th>Prediction</th>
                        <th>Target Price (T+1)</th>
                        <th>Actual Dir</th>
                        <th>PnL</th>
                    </tr>
                </thead>
                <tbody id="tableBody"></tbody>
            </table>
        </div>
        
        <div class="pagination">
            <button onclick="prevPage()">Previous</button>
            <span id="pageInfo"></span>
            <button onclick="nextPage()">Next</button>
        </div>
    </div>

    <script>
        // Data injected from Python
        const data = {json_data};
        const rowsPerPage = 20;
        let currentPage = 1;

        function renderTable() {{
            const tbody = document.getElementById('tableBody');
            tbody.innerHTML = '';

            const start = (currentPage - 1) * rowsPerPage;
            const end = start + rowsPerPage;
            const pageData = data.slice(start, end);

            pageData.forEach(row => {{
                const tr = document.createElement('tr');
                
                // Format prediction color
                const predClass = row.prediction === 'UP' ? 'up' : (row.prediction === 'DOWN' ? 'down' : '');
                const actualClass = row.actual === 'UP' ? 'up' : (row.actual === 'DOWN' ? 'down' : '');
                
                tr.innerHTML = `
                    <td>${{row.time_t}}</td>
                    <td>[${{row.price_t_2}}, ${{row.price_t_1}}, ${{row.price_t_rounded}}]</td>
                    <td>${{row.price_t.toFixed(2)}}</td>
                    <td class="${{predClass}}">${{row.prediction}}</td>
                    <td>${{row.next_price.toFixed(2)}}</td>
                    <td class="${{actualClass}}">${{row.actual}}</td>
                    <td style="color: ${{row.pnl >= 0 ? 'green' : 'red'}}">${{row.pnl.toFixed(2)}}</td>
                `;
                tbody.appendChild(tr);
            }});

            document.getElementById('pageInfo').innerText = `Page ${{currentPage}} of ${{Math.ceil(data.length / rowsPerPage)}}`;
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

        // Initial Render
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
        
        # Prepare data for JS (Converting timestamps to string for JSON serialization)
        # We limit the data passed to JS to avoid browser memory issues if millions of rows,
        # but for 5 years of 1h data (~50k rows), it's fine.
        js_data = []
        for r in test_results:
            js_data.append({
                'time_t': str(r['time_t']),
                'price_t_2': r['price_t_2'],
                'price_t_1': r['price_t_1'],
                'price_t_rounded': r['price_t_rounded'],
                'price_t': r['price_t'],
                'prediction': r['prediction'],
                'next_price': r['next_price'],
                'actual': r['actual'],
                'pnl': r['pnl']
            })
            
        import json
        json_str = json.dumps(js_data)
        
        html_content = HTML_TEMPLATE.format(
            symbol=SYMBOL,
            interval=INTERVAL,
            start=START_TIME,
            end=END_TIME,
            accuracy=accuracy,
            pnl=total_pnl,
            plot_data=plot_b64,
            json_data=json_str
        )
        
        self.wfile.write(html_content.encode('utf-8'))

# ---------------------------------------------------------
# Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    # 1. Fetch
    df = fetch_binance_data(SYMBOL, INTERVAL, START_TIME, END_TIME)
    
    if df.empty:
        print("No data fetched. Exiting.")
        sys.exit(1)
        
    # 2-5. Process, Train, Test
    train_df, test_df, test_results, accuracy, total_pnl = run_backtest(df)
    
    # 6. Generate Plot
    print("Generating plot...")
    plot_b64 = create_plot(df, test_results)
    
    # 7. Serve
    print(f"Starting server on port {PORT}...")
    print(f"Open your browser at http://localhost:{PORT}")
    
    with socketserver.TCPServer(("", PORT), BacktestHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")
            httpd.server_close()
