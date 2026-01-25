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
    # Pattern logic: We will use the DERIVATIVE columns of the 'close' price for the sequence pattern.
    # Using normalized raw prices for pattern matching is usually ineffective as price levels shift.
    pattern_map = {}

    # Build Training Dictionary
    # We iterate through the training data to build the "Knowledge Base"
    # Sequence is based on the 'd_close' of previous 3 candles
    values = train_df['d_close'].values
    
    # We need to look ahead to determine outcome
    # Outcome is based on the NEXT candle (open to close change) relative to threshold
    # However, prompt asks for next CLOSE rises/falls/stays. 
    # Usually this is (Close_next - Close_current) / Close_current
    
    # Pre-calculate next return for labeling
    # We use unrounded values for accurate labeling, but grouped by rounded sequences
    train_outcomes = (train_df['close'].shift(-1) - train_df['close']) / train_df['close']
    
    print("Training model...")
    for i in range(SEQ_LEN, len(train_df) - 1):
        # Sequence: last 3 derivative closes
        seq = tuple(values[i-SEQ_LEN+1:i+1]) # Tuple is hashable
        
        outcome_val = train_outcomes.iloc[i]
        
        if outcome_val > THRESHOLD:
            label = 'LONG'
        elif outcome_val < -THRESHOLD:
            label = 'SHORT'
        else:
            label = 'FLAT'
            
        if seq not in pattern_map:
            pattern_map[seq] = {'LONG': 0, 'SHORT': 0, 'FLAT': 0}
        
        pattern_map[seq][label] += 1

    # Convert counts to probabilities/predictions
    # We pick the outcome with the highest count
    prediction_rules = {}
    for seq, counts in pattern_map.items():
        best_outcome = max(counts, key=counts.get)
        # Only predict if there's a clear winner (optional, but good for stability)
        prediction_rules[seq] = best_outcome

    return prediction_rules, test_df

def run_backtest(prediction_rules, test_df):
    results = []
    
    values = test_df['d_close'].values
    opens = test_df['open'].values
    closes = test_df['close'].values
    
    # 8. Take test sequences and predict outcome
    # 9. Test accurate, pnl
    
    cumulative_pnl = 0.0
    wins = 0
    total_trades = 0
    
    # PnL History for plotting
    pnl_history = [0]
    dates = [test_df.index[0]]

    print("Running backtest...")
    # Iterate test data
    for i in range(SEQ_LEN, len(test_df) - 1):
        seq = tuple(values[i-SEQ_LEN+1:i+1])
        
        pred = prediction_rules.get(seq, 'FLAT') # Default to FLAT if unseen
        
        # Calculate actual PnL for the next candle
        # Prompt: "If we predict long we calculate pnl by (close-open)/open"
        # This implies we enter at Open of next candle and exit at Close of next candle
        
        entry_price = opens[i+1]
        exit_price = closes[i+1]
        
        trade_pnl = 0.0
        
        if pred == 'LONG':
            trade_pnl = (exit_price - entry_price) / entry_price
        elif pred == 'SHORT':
            trade_pnl = -(exit_price - entry_price) / entry_price
        
        # Check accuracy (Directional)
        actual_move = (exit_price - entry_price) / entry_price
        is_correct = False
        
        if pred == 'LONG' and actual_move > THRESHOLD: is_correct = True
        elif pred == 'SHORT' and actual_move < -THRESHOLD: is_correct = True
        elif pred == 'FLAT' and abs(actual_move) <= THRESHOLD: is_correct = True
        
        if pred != 'FLAT':
            cumulative_pnl += trade_pnl
            total_trades += 1
            if trade_pnl > 0:
                wins += 1
            
            pnl_history.append(cumulative_pnl)
            dates.append(test_df.index[i+1])
            
            results.append({
                'timestamp': test_df.index[i+1],
                'sequence': str(seq),
                'prediction': pred,
                'actual_move_pct': round(actual_move * 100, 2),
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
    plt.ylabel('PnL')
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
    report_data['table'] = results_df.tail(20).to_html(classes='table table-striped', index=False)

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
        <h1>ETH/USDT 1H Prediction Results</h1>
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
