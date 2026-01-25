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
START_DATE = '2024-01-01 00:00:00'
END_DATE = '2026-01-01 00:00:00'
SEQ_LEN = 4
ROUNDING_STEP = 0.001  # 0.2%
THRESHOLD = 0.01 
PORT = 8080

app = Flask(__name__)

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def fetch_and_prepare_data():
    ensure_dir(DATA_DIR)
    
    # 1. Force Clean Slate: Remove old file to ensure fresh fetch
    if os.path.exists(DATA_FILE):
        os.remove(DATA_FILE)
    
    print(f"Fetching fresh data for {SYMBOL} from Binance ({START_DATE} to {END_DATE})...")
    exchange = ccxt.binance({'enableRateLimit': True}) 
    
    since = int(pd.Timestamp(START_DATE).timestamp() * 1000)
    end_ts = int(pd.Timestamp(END_DATE).timestamp() * 1000)
    
    all_candles = []
    
    # --- A. FETCH RAW DATA ---
    while since < end_ts:
        try:
            ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, since=since, limit=1000)
            if not ohlcv: break
            
            all_candles.extend(ohlcv)
            last_timestamp = ohlcv[-1][0]
            since = last_timestamp + 1 
            
            if last_timestamp >= end_ts: break
            
            if len(all_candles) % 5000 == 0:
                print(f"Fetched up to {pd.to_datetime(last_timestamp, unit='ms')}")
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            break

    # 2. Create DataFrame and Set Timestamp as Index
    df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # !!! CRITICAL: Set Index to Timestamp. All subsequent column additions align to this. !!!
    df.set_index('timestamp', inplace=True)
    
    # Filter strictly within range
    df = df.sort_index()
    mask = (df.index >= pd.Timestamp(START_DATE)) & (df.index < pd.Timestamp(END_DATE))
    df = df.loc[mask]
    
    if df.empty:
        return df

    # --- B. ADD DERIVATIVE COLUMNS (Aligned by Timestamp) ---
    deriv_cols = ['d_open', 'd_high', 'd_low', 'd_close']
    for col in ['open', 'high', 'low', 'close']:
        df[f'd_{col}'] = df[col].pct_change().fillna(0)

    # --- C. ADD NORMALIZED COLUMNS (Aligned by Timestamp) ---
    first_vals = df.iloc[0]
    norm_cols = ['n_open', 'n_high', 'n_low', 'n_close']
    for col, n_col in zip(['open', 'high', 'low', 'close'], norm_cols):
        df[n_col] = df[col] / first_vals[col]

    # --- D. APPLY ROUNDING ---
    cols_to_round = deriv_cols + norm_cols
    df[cols_to_round] = (df[cols_to_round] / ROUNDING_STEP).round() * ROUNDING_STEP

    df.to_csv(DATA_FILE)
    print(f"Unified Data Saved: {len(df)} rows.")
    
    return df

def train_model(train_df, input_cols):
    pattern_map = {}
    data_values = train_df[input_cols].values
    train_outcomes = (train_df['close'].shift(-1) - train_df['close']) / train_df['close']
    
    for i in range(SEQ_LEN - 1, len(train_df) - 1):
        seq_array = data_values[i-SEQ_LEN+1 : i+1] 
        seq = tuple(seq_array.flatten()) 
        
        outcome_val = train_outcomes.iloc[i]
        
        if outcome_val > THRESHOLD: label = 'UP'
        elif outcome_val < -THRESHOLD: label = 'DOWN'
        else: label = 'FLAT'
            
        if seq not in pattern_map:
            pattern_map[seq] = {'UP': 0, 'DOWN': 0, 'FLAT': 0, 'total': 0}
        
        pattern_map[seq][label] += 1
        pattern_map[seq]['total'] += 1
        
    prediction_rules = {}
    sequence_stats = []

    for seq, counts in pattern_map.items():
        outcomes = {k: v for k, v in counts.items() if k != 'total'}
        best_outcome = max(outcomes, key=outcomes.get)
        prediction_rules[seq] = best_outcome
        
        sequence_stats.append({
            'sequence': seq,
            'total_count': counts['total'],
            'predicted_outcome': best_outcome,
            'up': counts['UP'],
            'down': counts['DOWN'],
            'flat': counts['FLAT']
        })
        
    return prediction_rules, sequence_stats

def build_sequences_and_predict(df):
    if len(df) < SEQ_LEN:
        return {}, {}, df, []

    # Use the unified DF for splitting
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    print("Training Derivative Model...")
    deriv_cols = ['d_open', 'd_high', 'd_low', 'd_close']
    deriv_rules, deriv_stats = train_model(train_df, deriv_cols)

    print("Training Raw Normalized Model...")
    norm_cols = ['n_open', 'n_high', 'n_low', 'n_close']
    norm_rules, norm_stats = train_model(train_df, norm_cols)

    deriv_stats.sort(key=lambda x: x['total_count'], reverse=True)
    top_10_sequences = deriv_stats[:10]

    return deriv_rules, norm_rules, test_df, top_10_sequences

def run_backtest(deriv_rules, norm_rules, test_df):
    results = []
    if len(test_df) < SEQ_LEN:
        return pd.DataFrame(), {}, [], []

    d_data = test_df[['d_open', 'd_high', 'd_low', 'd_close']].values
    n_data = test_df[['n_open', 'n_high', 'n_low', 'n_close']].values
    
    cumulative_pnl = 0.0
    wins = 0
    total_trades = 0
    
    pnl_history = [0]
    dates = [test_df.index[0]]

    print("Running backtest...")
    for i in range(SEQ_LEN - 1, len(test_df) - 1):
        
        # --- 1. PREDICTION LOGIC ---
        seq_array_d = d_data[i-SEQ_LEN+1 : i+1]
        seq_d = tuple(seq_array_d.flatten())
        pred_deriv = deriv_rules.get(seq_d, 'FLAT')

        seq_array_n = n_data[i-SEQ_LEN+1 : i+1]
        seq_n = tuple(seq_array_n.flatten())
        pred_norm = norm_rules.get(seq_n, 'FLAT')
        
        final_pred = 'FLAT'
        if (pred_deriv == 'UP' and pred_norm == 'DOWN') or \
           (pred_deriv == 'DOWN' and pred_norm == 'UP'):
            final_pred = 'FLAT' 
        else:
            if pred_deriv == 'UP' or pred_norm == 'UP':
                final_pred = 'UP'
            elif pred_deriv == 'DOWN' or pred_norm == 'DOWN':
                final_pred = 'DOWN'
            else:
                final_pred = 'FLAT'

        # --- 2. TRADE EXECUTION (Candle i+1) ---
        trade_candle = test_df.iloc[i+1]
        entry_price = trade_candle['open']
        exit_price = trade_candle['close']
        
        trade_pnl = 0.0
        if final_pred == 'UP':
            trade_pnl = (exit_price - entry_price) / entry_price
        elif final_pred == 'DOWN':
            trade_pnl = -(exit_price - entry_price) / entry_price
        
        if final_pred != 'FLAT':
            cumulative_pnl += trade_pnl
            total_trades += 1
            if trade_pnl > 0: wins += 1
            
            pnl_history.append(cumulative_pnl)
            dates.append(trade_candle.name) 
            
            # A. Input Sequence Candles
            input_candles_str = ""
            for k in range(SEQ_LEN):
                idx_loc = i - SEQ_LEN + 1 + k
                row = test_df.iloc[idx_loc]
                
                ts_str = row.name.strftime('%Y-%m-%d %H:%M')
                input_candles_str += f"[{k+1}] {ts_str} | O:{row['open']:.2f} H:{row['high']:.2f} L:{row['low']:.2f} C:{row['close']:.2f}<br>"

            # B. Target Candle
            trade_ts_str = trade_candle.name.strftime('%Y-%m-%d %H:%M')
            trade_ohlc_str = f"{trade_ts_str}<br>O:{trade_candle['open']:.2f} H:{trade_candle['high']:.2f} L:{trade_candle['low']:.2f} C:{trade_candle['close']:.2f}"
            
            results.append({
                'timestamp': trade_candle.name,
                'input_sequence_raw': input_candles_str,
                'trade_candle_raw': trade_ohlc_str,
                'pred_deriv': pred_deriv,
                'pred_norm': pred_norm,
                'final_prediction': final_pred,
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
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=False)
    
    norm_cols = ['n_open', 'n_high', 'n_low', 'n_close']
    plot_data = df[norm_cols]
    ax1.plot(plot_data.index, plot_data['n_close'], label='Norm Close', linewidth=1)
    ax1.set_title('Normalized OHLC (Close Price, Rounded)')
    ax1.legend()
    ax1.grid(True)

    deriv_cols = ['d_open', 'd_high', 'd_low', 'd_close']
    plot_deriv = df[deriv_cols]
    ax2.plot(plot_deriv.index, plot_deriv['d_close'], label='Deriv Close', color='orange', linewidth=0.5)
    ax2.set_title('Derivative OHLC (Close, Rounded)')
    ax2.legend()
    ax2.grid(True)

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

def generate_table_html(df_slice):
    """
    Helper to generate a consistent HTML table for a given dataframe slice.
    Includes a scrollable div wrapper.
    """
    html = '<div style="max-height: 500px; overflow-y: auto;">'
    html += '<table class="table table-bordered table-sm" style="font-size: 0.8rem; text-align: center;">'
    html += '<thead class="thead-light" style="position: sticky; top: 0; z-index: 1;">'
    html += '<tr>'
    html += '<th rowspan="2" style="vertical-align: middle;">Timestamp (UTC)</th>'
    html += '<th colspan="4">Raw Binance Data (USD)</th>'
    html += '<th colspan="4">Normalized (Rounded)</th>'
    html += '<th colspan="4">Derivative % (Rounded)</th>'
    html += '</tr>'
    html += '<tr>'
    html += '<th>Open</th><th>High</th><th>Low</th><th>Close</th>'
    html += '<th>O</th><th>H</th><th>L</th><th>C</th>'
    html += '<th>O</th><th>H</th><th>L</th><th>C</th>'
    html += '</tr>'
    html += '</thead>'
    html += '<tbody>'
    
    for index, row in df_slice.iterrows():
        ts_str = index.strftime('%Y-%m-%d %H:%M')
        r_vals = f"<td>{row['open']:.2f}</td><td>{row['high']:.2f}</td><td>{row['low']:.2f}</td><td>{row['close']:.2f}</td>"
        n_vals = f"<td>{row['n_open']:.4f}</td><td>{row['n_high']:.4f}</td><td>{row['n_low']:.4f}</td><td>{row['n_close']:.4f}</td>"
        d_vals = f"<td>{row['d_open']:.4f}</td><td>{row['d_high']:.4f}</td><td>{row['d_low']:.4f}</td><td>{row['d_close']:.4f}</td>"
        
        html += f"<tr><td>{ts_str}</td>{r_vals}{n_vals}{d_vals}</tr>"
        
    html += '</tbody></table></div>'
    return html

report_data = {}

def main_logic():
    df = fetch_and_prepare_data()
    if df.empty: 
        print("Error: No data fetched.")
        return

    # Filter for the specific range requested: Dec 29th 2025 to end of data (2026)
    # Using string slicing with pandas datetime index
    try:
        # Start from Dec 29, 2025. End is implicit (end of data)
        custom_slice = df.loc['2025-12-29':]
        report_data['custom_range'] = generate_table_html(custom_slice)
    except Exception as e:
        print(f"Error slicing data: {e}")
        report_data['custom_range'] = "<p>Error filtering data range.</p>"
    
    # Run predictions
    deriv_rules, norm_rules, test_df, top_seq = build_sequences_and_predict(df)
    results_df, stats, dates, pnl_curve = run_backtest(deriv_rules, norm_rules, test_df)
    
    plot_url = generate_charts(df, dates, pnl_curve)
    
    formatted_seq = []
    for item in top_seq:
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
        report_data['table'] = results_df.tail(20).to_html(classes='table table-striped table-sm', index=False, escape=False)
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
            td {{ vertical-align: middle !important; font-size: 0.85rem; }}
            td small {{ display: block; line-height: 1.2; }}
            table tr td:nth-child(2) {{ font-family: monospace; font-size: 0.75rem; white-space: nowrap; }}
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
            <div class="col-md-12">
                <h3>Unified Data Inspection</h3>
                <p>Showing data from <strong>Dec 29, 2025 through 2026</strong> (Raw, Normalized, Derivative).</p>
                {report_data.get('custom_range', '')}
            </div>
        </div>
        <hr>

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
                <h3>Top 10 Frequent Sequences (Derivative)</h3>
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
