import os
import ccxt
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Non-interactive backend for server environment
import matplotlib.pyplot as plt
import io
import base64
from flask import Flask, render_template_string

# Configuration
DATA_DIR = '/app/data'
DATA_FILE = os.path.join(DATA_DIR, 'ohlc.csv')
SYMBOL = 'ETH/USDT'
TIMEFRAME = '1h'
SEQ_LEN = 3
ROUNDING = 3  # 0.1% is 0.001, so 3 decimal places
THRESHOLD = 0.005 # 0.5%
PORT = 8080

app = Flask(__name__)

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def fetch_data():
    ensure_dir(DATA_DIR)
    
    if os.path.exists(DATA_FILE):
        print("Loading data from disk...")
        df = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)
    else:
        print("Fetching data from Binance...")
        exchange = ccxt.binance()
        # Fetching approx 1000 candles (default)
        ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df.to_csv(DATA_FILE)
    return df

def prepare_data(df):
    # 3. Create derivative version (price_i - price_i-1) / price_i-1
    # We apply this to Open, High, Low, Close
    # Since we can't do it for the first row, we fill with 0
    deriv_cols = ['d_open', 'd_high', 'd_low', 'd_close']
    for col in ['open', 'high', 'low', 'close']:
        df[f'd_{col}'] = df[col].pct_change().fillna(0)

    # 5. Divide raw ohlc by first candle (e.g. open / first open)
    # We normalize each column by the very first value of that column in the dataset
    first_vals = df.iloc[0]
    norm_cols = ['n_open', 'n_high', 'n_low', 'n_close']
    for col, n_col in zip(['open', 'high', 'low', 'close'], norm_cols):
        df[n_col] = df[col] / first_vals[col]

    # 6. Round raw ohlc (normalized) and derivative version to 0.1% (3 decimal places)
    # Note: 0.1% = 0.001
    cols_to_round = deriv_cols + norm_cols
    df[cols_to_round] = df[cols_to_round].round(ROUNDING)

    return df

def build_sequences_and_predict(df):
    # Split Train/Test (80/20)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    # 7. Count outcome of every sequence
    pattern_map = {}

    # We extract the derivative columns as a numpy array for speed/slicing
    # Columns: d_open, d_high, d_low, d_close
    d_data = train_df[['d_open', 'd_high', 'd_low', 'd_close']].values
    
    # Calculate returns for labeling (Close to Close change of the NEXT candle)
    # Using raw close prices for accuracy
    train_outcomes = (train_df['close'].shift(-1) - train_df['close']) / train_df['close']
    
    print("Training model on OHLC sequences...")
    # Iterate through training data
    # i represents the index of the 'current' candle (end of sequence)
    for i in range(SEQ_LEN - 1, len(train_df) - 1):
        # Sequence: last 3 candles of OHLC derivatives
        # Slice is exclusive on upper bound, so i+1 includes row i
        seq_array = d_data[i-SEQ_LEN+1 : i+1] 
        seq = tuple(seq_array.flatten()) # Flatten 3x4 array to 1D tuple for hashing
        
        outcome_val = train_outcomes.iloc[i]
        
        # Determine Label
        if outcome_val > THRESHOLD:
            label = 'UP'
        elif outcome_val < -THRESHOLD:
            label = 'DOWN'
        else:
            label = 'FLAT'
            
        if seq not in pattern_map:
            pattern_map[seq] = {'UP': 0, 'DOWN': 0, 'FLAT': 0}
        
        pattern_map[seq][label] += 1

    # Convert counts to prediction rules (Winner takes all)
    prediction_rules = {}
    for seq, counts in pattern_map.items():
        best_outcome = max(counts, key=counts.get)
        prediction_rules[seq] = best_outcome

    return prediction_rules, test_df

def run_backtest(prediction_rules, test_df):
    results = []
    
    # Prepare test data arrays
    d_data = test_df[['d_open', 'd_high', 'd_low', 'd_close']].values
    opens = test_df['open'].values
    closes = test_df['close'].values
    
    cumulative_pnl = 0.0
    wins = 0
    total_trades = 0
    
    # PnL History for plotting
    pnl_history = [0]
    dates = [test_df.index[0]]

    print("Running backtest...")
    # Iterate test data
    for i in range(SEQ_LEN - 1, len(test_df) - 1):
        seq_array = d_data[i-SEQ_LEN+1 : i+1]
        seq = tuple(seq_array.flatten())
        
        pred = prediction_rules.get(seq, 'FLAT') # Default to FLAT if unseen
        
        # Trade execution on NEXT candle
        entry_price = opens[i+1]
        exit_price = closes[i+1]
        
        trade_pnl = 0.0
        
        # Calculate PnL based on prediction
        if pred == 'UP':
            trade_pnl = (exit_price - entry_price) / entry_price
        elif pred == 'DOWN':
            trade_pnl = -(exit_price - entry_price) / entry_price
        
        # Check accuracy (Directional)
        actual_move_pct = (exit_price - entry_price) / entry_price
        
        if pred != 'FLAT':
            cumulative_pnl += trade_pnl
            total_trades += 1
            if trade_pnl > 0:
                wins += 1
            
            pnl_history.append(cumulative_pnl)
            dates.append(test_df.index[i+1])
            
            results.append({
                'timestamp': test_df.index[i+1],
                'prediction': pred,
                'actual_move_pct': round(actual_move_pct * 100, 2),
                'pnl': round(trade_pnl * 100, 2),
                'cum_pnl': round(cumulative_pnl * 100, 2)
            })

    stats = {
        'total_trades': total_trades,
        'win_rate': round((wins / total_trades * 100), 2) if total_trades > 0 else 0,
        'final_pnl_pct': round(cumulative_pnl * 100, 2)
    }
    
    return pd.DataFrame(results), stats, dates, pnl_history

def generate_plot(dates, pnl_history):
    plt.figure(figsize=(10, 5))
    plt.plot(dates, pnl_history, label='Strategy PnL (Cumulative %)')
    plt.title(f'Backtest Results: ETH/USDT ({TIMEFRAME})')
    plt.ylabel('PnL (Decimal)')
    plt.xlabel('Date')
    plt.legend()
    plt.grid(True)
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

# Global storage for server
report_data = {}

def main_logic():
    df = fetch_data()
    df = prepare_data(df)
    rules, test_df = build_sequences_and_predict(df)
    results_df, stats, dates, pnl_curve = run_backtest(rules, test_df)
    
    plot_url = generate_plot(dates, pnl_curve)
    
    # Store for Flask
    report_data['stats'] = stats
    report_data['plot'] = plot_url
    if not results_df.empty:
        report_data['table'] = results_df.tail(20).to_html(classes='table table-striped', index=False)
    else:
        report_data['table'] = "<p>No trades executed in test set.</p>"

@app.route('/')
def home():
    if not report_data:
        return "Processing data... please refresh in a moment."
    
    html = f"""
    <html>
    <head>
        <title>Crypto Prediction Bot</title>
        <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css">
        <style>body{{ padding: 20px; }}</style>
    </head>
    <body>
        <h1>ETH/USDT 1H OHLC Prediction</h1>
        <hr>
        <div class="row">
            <div class="col-md-4">
                <h3>Performance Stats</h3>
                <ul class="list-group">
                    <li class="list-group-item">Total Trades: {report_data['stats']['total_trades']}</li>
                    <li class="list-group-item">Win Rate: {report_data['stats']['win_rate']}%</li>
                    <li class="list-group-item">Final PnL: {report_data['stats']['final_pnl_pct']}%</li>
                </ul>
            </div>
            <div class="col-md-8">
                <img src="data:image/png;base64,{report_data['plot']}" style="width:100%">
            </div>
        </div>
        <hr>
        <h3>Recent Test Trades (Last 20)</h3>
        {report_data['table']}
    </body>
    </html>
    """
    return render_template_string(html)

if __name__ == "__main__":
    try:
        main_logic()
        print(f"Server starting on port {PORT}...")
        app.run(host='0.0.0.0', port=PORT)
    except Exception as e:
        print(f"Error: {e}")
