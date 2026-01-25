import os
import ccxt
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import time
from flask import Flask, render_template_string
from sklearn.neighbors import RadiusNeighborsClassifier

# --- Configuration ---
DATA_PATH = '/app/data/ohlc_full.csv'
MODEL_PATH = '/app/data/painting_model_full.pkl'
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1h'
TOTAL_CANDLES = 10000  # "Remove sample limitations" - fetched via pagination
THRESHOLD = 0.001
SPHERE_RADIUS = 2.0 
PORT = 8080

app = Flask(__name__)

# --- Custom Fading Sphere Logic ---
def fading_sphere_kernel(distances):
    weights = np.empty(len(distances), dtype=object)
    for i, d in enumerate(distances):
        if len(d) > 0:
            weights[i] = np.maximum(0.0, 1.0 - (d / SPHERE_RADIUS))
        else:
            weights[i] = np.array([])
    return weights

# --- 1. Fetch Data (Pagination Enabled) ---
def get_ohlc_data():
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    
    # If file exists and is large enough, load it
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        if len(df) >= TOTAL_CANDLES * 0.9:
            print(f"Loading {len(df)} candles from disk...")
            return df
    
    print(f"Fetching ~{TOTAL_CANDLES} candles from Binance...")
    exchange = ccxt.binance()
    
    # Calculate start time for 10,000 hours ago
    since = exchange.milliseconds() - (TOTAL_CANDLES * 60 * 60 * 1000)
    all_ohlc = []
    
    while len(all_ohlc) < TOTAL_CANDLES:
        try:
            ohlc = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, since=since, limit=1000)
            if not ohlc:
                break
            since = ohlc[-1][0] + 1  # Move to next ms
            all_ohlc += ohlc
            print(f"Fetched {len(all_ohlc)} candles...")
            time.sleep(0.1) # Respect rate limits
        except Exception as e:
            print(f"Fetch error: {e}")
            break
            
    df = pd.DataFrame(all_ohlc, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df.drop_duplicates(subset=['timestamp'], inplace=True)
    df.to_csv(DATA_PATH, index=False)
    return df

# --- 2. Process & Feature Engineering ---
def prepare_data(df):
    raw_ohlc = df[['open', 'high', 'low', 'close']].values
    # Derivative OHLC (first entry 0)
    derivative = np.diff(raw_ohlc, axis=0, prepend=0)
    
    X = []
    y = []
    y_raw_return = []
    input_verification = [] # Store raw close prices for verification
    
    seq_len = 3
    
    # Create 12D points
    for i in range(len(derivative) - seq_len):
        # Feature: 12 dims
        X.append(derivative[i:i+seq_len].flatten())
        
        # Verification Data: The Raw Close prices of the 3 candles in sequence
        # Indices: i, i+1, i+2
        closes = df['close'].values[i:i+seq_len]
        input_verification.append(closes)
        
        # Target: 4th candle direction (Index i+3 relative to start, or i+seq_len)
        # We predict movement from end of sequence (i+seq_len-1) to next (i+seq_len)
        curr_price = df['close'].values[i+seq_len-1]
        next_price = df['close'].values[i+seq_len]
        
        change = (next_price - curr_price) / curr_price
        y_raw_return.append(change)
        
        if change > THRESHOLD: y.append(1.0)
        elif change < -THRESHOLD: y.append(-1.0)
        else: y.append(0.0)
            
    return np.array(X), np.array(y), np.array(y_raw_return), np.array(input_verification)

# --- 3. Training ---
def train_painting(X_train, y_train):
    print("Painting the canvas (Building Spatial Tree)...")
    clf = RadiusNeighborsClassifier(radius=SPHERE_RADIUS, 
                                    weights=fading_sphere_kernel, 
                                    algorithm='auto', 
                                    outlier_label=0)
    clf.fit(X_train, y_train)
    return clf

# --- 4. Plotting ---
def create_plots(y_true, y_pred, returns):
    # Plot 1: Cumulative PnL (Full History)
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    cumulative_returns = np.cumsum(returns) * 100
    
    # Create a gradient-like fill or just a clean line
    ax1.plot(cumulative_returns, color='#00ff88', linewidth=1.5, label='Strategy')
    ax1.fill_between(range(len(cumulative_returns)), cumulative_returns, 0, color='#00ff88', alpha=0.1)
    
    # Add a baseline (Hold)
    # ax1.plot(np.cumsum(y_raw_test) * 100, color='gray', alpha=0.5, label='Buy & Hold')
    
    ax1.set_facecolor('#1e1e1e')
    fig1.patch.set_facecolor('#1e1e1e')
    ax1.tick_params(colors='white')
    ax1.set_title(f'Strategy PnL over {len(returns)} Trades', color='white', fontsize=14)
    ax1.set_ylabel('Return (%)', color='white')
    ax1.set_xlabel('Trade Sequence #', color='white')
    ax1.grid(True, alpha=0.1)
    ax1.legend()
    
    buf1 = io.BytesIO()
    fig1.savefig(buf1, format='png', bbox_inches='tight')
    plot_pnl = base64.b64encode(buf1.getvalue()).decode()
    plt.close(fig1)
    
    # Plot 2: Confusion Matrix
    from sklearn.metrics import confusion_matrix
    try:
        cm = confusion_matrix(y_true, y_pred, labels=[-1, 0, 1])
        fig2, ax2 = plt.subplots(figsize=(5, 5))
        ax2.imshow(cm, cmap='Blues')
        ax2.set_title('Confusion Matrix')
        ax2.set_xticklabels(['Short', 'Flat', 'Long'])
        ax2.set_yticklabels(['Short', 'Flat', 'Long'])
        
        for i in range(3):
            for j in range(3):
                ax2.text(j, i, cm[i, j], ha='center', va='center', 
                         color='white' if cm[i,j] > cm.max()/2 else 'black')
        
        buf2 = io.BytesIO()
        fig2.savefig(buf2, format='png', bbox_inches='tight')
        plot_cm = base64.b64encode(buf2.getvalue()).decode()
        plt.close(fig2)
    except:
        plot_cm = ""

    return plot_pnl, plot_cm

# --- Main Logic ---
RESULTS = {}

def update_system():
    # Cleanup old models
    if os.path.exists(MODEL_PATH):
        try:
            joblib.load(MODEL_PATH)
        except:
            os.remove(MODEL_PATH)

    # 1. Get Data
    df = get_ohlc_data()
    X, y, y_raw, inputs = prepare_data(df)
    
    # 2. Split (70/30)
    split = int(len(X) * 0.7)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    y_raw_test = y_raw[split:]
    inputs_test = inputs[split:]
    
    # 3. Normalize
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std
    
    # 4. Load/Train
    if os.path.exists(MODEL_PATH):
        print("Loading painting...")
        clf = joblib.load(MODEL_PATH)
    else:
        clf = train_painting(X_train_norm, y_train)
        print("Saving painting...")
        joblib.dump(clf, MODEL_PATH)
        
    # 5. Inference
    print(f"Inference on {len(X_test)} points...")
    preds = clf.predict(X_test_norm)
    
    # 6. Stats & Assets
    acc = np.mean(preds == y_test) * 100
    strat_returns = preds * y_raw_test
    total_pnl = np.sum(strat_returns) * 100
    
    plot_pnl, plot_cm = create_plots(y_test, preds, strat_returns)
    
    # 7. Create Verification Table
    # We combine Inputs (Candle 1, 2, 3) with Prediction and Result
    df_res = pd.DataFrame({
        'C1': inputs_test[:, 0],
        'C2': inputs_test[:, 1],
        'C3': inputs_test[:, 2],
        'Pred_Dir': preds,
        'Actual_Dir': y_test,
        'Actual_Ret': y_raw_test,
        'Strat_PnL': strat_returns
    })
    
    RESULTS['acc'] = f"{acc:.2f}"
    RESULTS['pnl'] = f"{total_pnl:.2f}"
    RESULTS['count'] = len(X_test)
    RESULTS['plot_pnl'] = plot_pnl
    RESULTS['plot_cm'] = plot_cm
    # Showing last 20 rows
    RESULTS['table'] = df_res.tail(20).to_html(classes='table', index=False, float_format='%.2f')

# --- Web Server ---
HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>13D Sphere Strategy</title>
    <style>
        body { background-color: #121212; color: #e0e0e0; font-family: 'Segoe UI', monospace; padding: 20px; margin: 0; }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 { color: #00ff88; text-transform: uppercase; letter-spacing: 2px; border-bottom: 1px solid #333; padding-bottom: 10px; }
        .metrics { display: flex; gap: 20px; margin-bottom: 20px; }
        .metric-box { background: #1e1e1e; padding: 15px; border-radius: 5px; flex: 1; text-align: center; border: 1px solid #333; }
        .metric-val { font-size: 1.5em; font-weight: bold; color: #fff; }
        .metric-lbl { color: #888; font-size: 0.9em; }
        .charts { display: flex; flex-wrap: wrap; gap: 20px; margin-bottom: 30px; }
        .chart-main { flex: 2; min-width: 300px; }
        .chart-sub { flex: 1; min-width: 300px; display: flex; align-items: center; justify-content: center; background: #1e1e1e; border-radius: 5px; }
        img { max-width: 100%; height: auto; border-radius: 5px; }
        table { width: 100%; border-collapse: collapse; background: #1e1e1e; font-size: 0.9em; }
        th { text-align: left; padding: 12px; background: #252525; color: #00ff88; }
        td { padding: 10px; border-bottom: 1px solid #333; }
        tr:hover { background: #2a2a2a; }
        .pos { color: #00ff88; }
        .neg { color: #ff4444; }
    </style>
</head>
<body>
    <div class="container">
        <h1>13-Dimensional Sphere Inference</h1>
        
        <div class="metrics">
            <div class="metric-box">
                <div class="metric-val">{{ r.acc }}%</div>
                <div class="metric-lbl">Accuracy</div>
            </div>
            <div class="metric-box">
                <div class="metric-val" style="color: {{ 'red' if r.pnl.startswith('-') else '#00ff88' }}">{{ r.pnl }}%</div>
                <div class="metric-lbl">Total PnL</div>
            </div>
            <div class="metric-box">
                <div class="metric-val">{{ r.count }}</div>
                <div class="metric-lbl">Test Samples</div>
            </div>
        </div>

        <div class="charts">
            <div class="chart-main">
                <img src="data:image/png;base64,{{ r.plot_pnl }}">
            </div>
            <div class="chart-sub">
                <img src="data:image/png;base64,{{ r.plot_cm }}">
            </div>
        </div>

        <h3>Verification: Recent Input Sequences & Predictions</h3>
        <p style="color:#888; font-size:0.8em">C1, C2, C3 are the raw closing prices of the input sequence. Pred_Dir is the strategy's guess (1=Up, -1=Down, 0=Flat).</p>
        <div style="overflow-x: auto;">
            {{ r.table|safe }}
        </div>
    </div>
</body>
</html>
"""

@app.route('/')
def home():
    if not RESULTS:
        return "<body style='background:#121212;color:white'><h1>System Initializing...</h1><p>Fetching 10k candles and painting dimensions. Please refresh in 30 seconds.</p></body>"
    return render_template_string(HTML, r=RESULTS)

if __name__ == '__main__':
    print("Starting pipeline...")
    update_system()
    app.run(host='0.0.0.0', port=PORT)
