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
DATA_FILE = os.path.join(DATA_DIR, 'ohlc_2025.csv')
SYMBOL = 'ETH/USDT'
TIMEFRAME = '1h'
START_DATE = '2025-01-01 00:00:00'
END_DATE = '2026-01-01 00:00:00'
SEQ_LEN = 3
ROUNDING_STEP = 0.002  # 0.2%
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
        print(f"Fetching data from Binance ({START_DATE} to {END_DATE})...")
        exchange = ccxt.binance()
        
        since = int(pd.Timestamp(START_DATE).timestamp() * 1000)
        end_ts = int(pd.Timestamp(END_DATE).timestamp() * 1000)
        
        all_candles = []
        
        while since < end_ts:
            try:
                ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, since=since, limit=1000)
                if not ohlcv: break
                
                all_candles.extend(ohlcv)
                last_timestamp = ohlcv[-1][0]
                since = last_timestamp + 1 
                
                if last_timestamp >= end_ts: break
                print(f"Fetched up to {pd.to_datetime(last_timestamp, unit='ms')}")
                
            except Exception as e:
                print(f"Error fetching data: {e}")
                break

        df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        mask = (df.index >= pd.Timestamp(START_DATE)) & (df.index < pd.Timestamp(END_DATE))
        df = df.loc[mask]
        
        df.to_csv(DATA_FILE)
        print(f"Saved {len(df)} rows to {DATA_FILE}")
        
    return df

def prepare_data(df):
    # 3. Create derivative version
    deriv_cols = ['d_open', 'd_high', 'd_low', 'd_close']
    for col in ['open', 'high', 'low', 'close']:
        df[f'd_{col}'] = df[col].pct_change().fillna(0)

    # 5. Divide raw ohlc by first candle (Normalize)
    if len(df) > 0:
        first_vals = df.iloc[0]
        norm_cols = ['n_open', 'n_high', 'n_low', 'n_close']
        for col, n_col in zip(['open', 'high', 'low', 'close'], norm_cols):
            df[n_col] = df[col] / first_vals[col]

        # 6. Round raw ohlc (normalized) and derivative version to 0.2%
        cols_to_round = deriv_cols + norm_cols
        df[cols_to_round] = (df[cols_to_round] / ROUNDING_STEP).round() * ROUNDING_STEP

    return df

def build_sequences_and_predict(df):
    if len(df) < SEQ_LEN:
        return {}, df, []

    # Split Train/Test
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    # 7. Count outcome of every sequence
    pattern_map = {}

    d_data = train_df[['d_open', 'd_high', 'd_low', 'd_close']].values
    train_outcomes = (train_df['close'].shift(-1) - train_df['close']) / train_df['close']
    
    print("Training model on OHLC sequences...")
    for i in range(SEQ_LEN - 1, len(train_df) - 1):
        seq_array = d_data[i-SEQ_LEN+1 : i+1] 
        seq = tuple(seq_array.flatten()) 
        
        outcome_val = train_outcomes.iloc[i]
        
        if outcome_val > THRESHOLD: label = 'UP'
        elif outcome_val < -THRESHOLD: label = 'DOWN'
        else: label = 'FLAT'
            
        if seq not in pattern_map:
            pattern_map[seq] = {'UP': 0, 'DOWN': 0, 'FLAT': 0, 'total': 0}
        
        pattern_map[seq][label] += 1
        pattern_map[seq]['total'] += 1

    # Generate Prediction Rules and Top Sequences Stats
    prediction_rules = {}
    sequence_stats = []

    for seq, counts in pattern_map.items():
        # Determine best outcome (excluding 'total' key)
        outcomes = {k: v for k, v in counts.items() if k != 'total'}
        best_outcome = max(outcomes, key=outcomes.get)
        prediction_rules[seq] = best_outcome
        
        # Store for top 10 table
        sequence_stats.append({
            'sequence': seq,
            'total_count': counts['total'],
            'predicted_outcome': best_outcome,
            'up': counts['UP'],
            'down': counts['DOWN'],
            'flat': counts['FLAT']
        })

    # Sort by total frequency descending
    sequence_stats.sort(key=lambda x: x['total_count'], reverse=True)
    top_10_sequences = sequence_stats[:10]

    return prediction_rules, test_df, top_10_sequences

def run_backtest(prediction_rules, test_df):
    results = []
    if len(test_df) < SEQ_LEN:
        return pd.DataFrame(), {}, [], []

    d_data = test_df[['d_open', 'd_high', 'd_low', 'd_close']].values
    opens = test_df['open'].values
    closes = test_df['close'].values
    
    cumulative_pnl = 0.0
    wins = 0
    total_trades = 0
    
    pnl_history = [0]
    dates = [test_df.index[0]]

    print("Running backtest...")
    for i in range(SEQ_LEN - 1, len(test_df) - 1):
        seq_array = d_data[i-SEQ_LEN+1 : i+1]
        seq = tuple(seq_array.flatten())
        
        pred = prediction_rules.get(seq, 'FLAT')
        
        entry_price = opens[i+1]
        exit_price = closes[i+1]
        
        trade_pnl = 0.0
        
        if pred == 'UP':
            trade_pnl = (exit_price - entry_price) / entry_price
        elif pred == 'DOWN':
            trade_pnl = -(exit_price - entry_price) / entry_price
        
        if pred != 'FLAT':
            cumulative_pnl += trade_pnl
            total_trades += 1
            if trade_pnl > 0: wins += 1
            
            pnl_history.append(cumulative_pnl)
            dates.append(test_df.index[i+1])
            
            results.append({
                'timestamp': test_df.index[i+1],
                'prediction': pred,
                'pnl': round(trade_pnl * 100, 2),
                'cum_pnl': round(cumulative_pnl * 100, 2)
            })

    stats = {
        'total_trades': total_trades,
        'win_rate': round((wins / total_trades * 100), 2) if total_trades > 0 else 0,
        'final_pnl_pct': round(cumulative_pnl * 100, 2)
    }
    
    return pd.DataFrame(results), stats, dates, pnl_history

def generate_charts(df, pnl_dates, pnl_history):
    # Create a figure with 3 subplots: Normalized, Derivative, PnL
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=False)
    
    # 1. Normalized Raw OHLC (Rounded)
    norm_cols = ['n_open', 'n_high', 'n_low', 'n_close']
    # Plotting only the last 500 points for clarity if dataset is large, or full if small
    # For full context we plot all, but with thin lines
    plot_data = df[norm_cols]
    ax1.plot(plot_data.index, plot_data['n_close'], label='Norm Close', linewidth=1)
    ax1.set_title('Normalized OHLC (Close Price, Rounded)')
    ax1.legend()
    ax1.grid(True)

    # 2. Derivative OHLC (Rounded)
    deriv_cols = ['d_open', 'd_high', 'd_low', 'd_close']
    plot_deriv = df[deriv_cols]
    ax2.plot(plot_deriv.index, plot_deriv['d_close'], label='Deriv Close', color='orange', linewidth=0.5)
    ax2.set_title('Derivative OHLC (Close, Rounded)')
    ax2.legend()
    ax2.grid(True)

    # 3. Strategy PnL
    if len(pnl_dates) > 1:
        ax3.plot(pnl_dates, pnl_history, label='Strategy PnL', color='green')
    else:
        ax3.text(0.5, 0.5, 'Not enough trades', ha='center')
    ax3.set_title(f'Strategy PnL ({START_DATE} - {END_DATE})')
    ax3.set_ylabel('PnL (Decimal)')
    ax3.grid(True)

    plt.tight_layout()
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    return base64.b64encode(img.getvalue()).decode()

report_data = {}

def main_logic():
    df = fetch_data()
    if df.empty: return

    df = prepare_data(df)
    rules, test_df, top_seq = build_sequences_and_predict(df)
    results_df, stats, dates, pnl_curve = run_backtest(rules, test_df)
    
    plot_url = generate_charts(df, dates, pnl_curve)
    
    # Format Top 10 Sequence Table
    # Convert tuple sequences to string for display
    formatted_seq = []
    for item in top_seq:
        # Format the numbers in the sequence for readability
        seq_str = ', '.join([f"{x:.3f}" for x in item['sequence']])
        formatted_seq.append({
            'sequence': f"<small>{seq_str}</small>",
            'count': item['total_count'],
            'prediction': item['predicted_outcome'],
            'breakdown': f"U:{item['up']} D:{item['down']} F:{item['flat']}"
        })
    
    seq_df = pd.DataFrame(formatted_seq)

    report_data['stats'] = stats
    report_data['plot'] = plot_url
    report_data['top_sequences'] = seq_df.to_html(classes='table table-bordered table-sm', escape=False, index=False)
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
        <style>
            body{{ padding: 20px; }}
            .chart-container {{ margin-bottom: 30px; }}
        </style>
    </head>
    <body>
        <h1>{SYMBOL} Analysis ({START_DATE} - {END_DATE})</h1>
        <hr>
        
        <div class="row">
            <div class="col-md-12 chart-container">
                 
                <img src="data:image/png;base64,{report_data.get('plot', '')}" style="width:100%">
            </div>
        </div>

        <div class="row">
            <div class="col-md-4">
                <h3>Performance Stats</h3>
                <ul class="list-group">
                    <li class="list-group-item">Total Trades: {report_data.get('stats', {}).get('total_trades', 0)}</li>
                    <li class="list-group-item">Win Rate: {report_data.get('stats', {}).get('win_rate', 0)}%</li>
                    <li class="list-group-item">Final PnL: {report_data.get('stats', {}).get('final_pnl_pct', 0)}%</li>
                </ul>
            </div>
            
            <div class="col-md-8">
                <h3>Top 10 Frequent Sequences</h3>
                {report_data.get('top_sequences', '')}
            </div>
        </div>

        <hr>
        <h3>Recent Test Trades (Last 20)</h3>
        {report_data.get('table', '')}
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
