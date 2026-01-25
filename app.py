import os
import ccxt
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server usage
import matplotlib.pyplot as plt
import io
import base64
from flask import Flask, render_template_string

# --- Configuration ---
DATA_PATH = '/app/data/ohlc.csv'
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1h'
LIMIT = 2000  # Number of candles to fetch
THRESHOLD = 0.001  # Threshold for "0" direction (0.1% move)
SPHERE_RADIUS = 0.05  # The radius of the fading sphere. Needs to be tuned to data scale.
PORT = 8080

app = Flask(__name__)

# --- 1. Fetch & 2/3. Save/Load ---
def get_ohlc_data():
    # Ensure directory exists
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)

    if os.path.exists(DATA_PATH):
        print("Loading data from disk...")
        df = pd.read_csv(DATA_PATH)
    else:
        print("Fetching data from Binance...")
        exchange = ccxt.binance()
        try:
            ohlc = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=LIMIT)
            df = pd.DataFrame(ohlc, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df.to_csv(DATA_PATH, index=False)
        except Exception as e:
            return None, f"Error fetching data: {str(e)}"
    
    return df, None

# --- 4. Processing ---
def process_data(df):
    raw_ohlc = df[['open', 'high', 'low', 'close']].values
    
    # Derivative OHLC (Change from previous candle)
    # Assume first entry is 0 to match dimensions
    derivative_ohlc = np.diff(raw_ohlc, axis=0, prepend=0)
    
    # We use derivative data for features to make them stationary (comparable over time)
    # normalize derivative data for the "sphere" logic to work better
    # Simple min-max or std scaling is usually needed for distance-based algos.
    # However, to strictly follow "raw/derivative" without external scaler libraries, 
    # we proceed with derivative_ohlc.
    
    return raw_ohlc, derivative_ohlc

# --- 6. Sequences & 7. Labeling ---
def create_dataset(data, raw_close, seq_length=3):
    X = []
    y = []
    y_raw_change = [] # To calc PnL later

    # We need 3 candles for X, and the 4th candle for y
    for i in range(len(data) - seq_length):
        # Feature: Flatten 3 candles * 4 dims = 12 dims
        seq = data[i:i+seq_length].flatten()
        X.append(seq)
        
        # Target: 4th candle direction
        # We compare Close of (i+seq_length) vs Close of (i+seq_length-1)
        current_close = raw_close[i+seq_length-1]
        next_close = raw_close[i+seq_length]
        
        change_pct = (next_close - current_close) / current_close
        y_raw_change.append(change_pct)
        
        if change_pct > THRESHOLD:
            label = 1.0
        elif change_pct < -THRESHOLD:
            label = -1.0
        else:
            label = 0.0
        y.append(label)

    return np.array(X), np.array(y), np.array(y_raw_change)

# --- 8 & 9. The Painting Algorithm ---
def infer_painting(X_train, y_train, X_test):
    """
    Implements the 13-dimensional painting logic.
    """
    # Construct the 13-dimensional Training Points
    # X_train is (N, 12), y_train is (N,). We stack them to (N, 13)
    train_points_13d = np.hstack((X_train, y_train.reshape(-1, 1)))
    
    predictions = []
    
    # Possible directions for the 13th dimension
    candidates = [-1.0, 0.0, 1.0]
    
    print(f"Running inference on {len(X_test)} points. This might take a moment...")
    
    # For every point in test set
    for i, x_t in enumerate(X_test):
        best_intensity = -1
        best_dir = 0
        
        # We test which "height" (direction) in the 13th dimension has the most "paint"
        for cand_dir in candidates:
            # Construct the test point in 13D space with the hypothesis direction
            test_point_13d = np.append(x_t, cand_dir)
            
            # Vectorized Euclidean distance calculation
            # Dist between this test hypothesis and ALL training spheres
            dists = np.linalg.norm(train_points_13d - test_point_13d, axis=1)
            
            # Fading sphere function: 1 in middle, decreases to 0. 
            # Logic: Intensity = max(0, 1 - (dist / RADIUS))
            # If dist > RADIUS, intensity is 0.
            
            # We scale distance to make the radius effective.
            intensities = np.maximum(0, 1 - (dists / SPHERE_RADIUS))
            
            total_intensity = np.sum(intensities)
            
            if total_intensity > best_intensity:
                best_intensity = total_intensity
                best_dir = cand_dir
        
        predictions.append(best_dir)
        
    return np.array(predictions)

# --- Plotting Utility ---
def create_plots(y_test, predictions, pnl_curve):
    # Plot 1: PnL Curve
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(np.cumsum(pnl_curve), label='Cumulative PnL (%)', color='green')
    ax1.set_title('Strategy PnL over Test Set')
    ax1.set_xlabel('Trade #')
    ax1.set_ylabel('Return')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    img1 = io.BytesIO()
    fig1.savefig(img1, format='png')
    img1.seek(0)
    plot_url1 = base64.b64encode(img1.getvalue()).decode()
    plt.close(fig1)

    # Plot 2: Confusion Matrix / Distribution
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    # Simple scatter or bar chart of predictions vs actual
    from sklearn.metrics import confusion_matrix
    try:
        cm = confusion_matrix(y_test, predictions, labels=[-1, 0, 1])
        im = ax2.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax2.figure.colorbar(im, ax=ax2)
        ax2.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
               xticklabels=[-1, 0, 1], yticklabels=[-1, 0, 1],
               title='Confusion Matrix', ylabel='True Label', xlabel='Predicted Label')
    except:
        ax2.text(0.5, 0.5, "Insufficient data for Matrix", ha='center')
    
    img2 = io.BytesIO()
    fig2.savefig(img2, format='png')
    img2.seek(0)
    plot_url2 = base64.b64encode(img2.getvalue()).decode()
    plt.close(fig2)

    return plot_url1, plot_url2

# --- HTML Template ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>13D Painting Trading Strategy</title>
    <style>
        body { font-family: sans-serif; margin: 2rem; background: #f0f2f5; }
        .container { max-width: 1000px; margin: 0 auto; background: white; padding: 2rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        h1 { color: #333; }
        .stats-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-bottom: 2rem; }
        .stat-box { background: #f8f9fa; padding: 1rem; border-radius: 4px; text-align: center; border: 1px solid #dee2e6; }
        .stat-val { font-size: 1.5rem; font-weight: bold; color: #007bff; }
        .plots { display: flex; flex-wrap: wrap; gap: 2rem; justify-content: center; }
        img { max-width: 100%; height: auto; border: 1px solid #ddd; }
        table { width: 100%; border-collapse: collapse; margin-top: 2rem; }
        th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <div class="container">
        <h1>13-Dimensional "Sphere" Inference Results</h1>
        
        <div class="stats-grid">
            <div class="stat-box">
                <div>Accuracy</div>
                <div class="stat-val">{{ accuracy }}%</div>
            </div>
            <div class="stat-box">
                <div>Total PnL</div>
                <div class="stat-val">{{ total_pnl }}%</div>
            </div>
            <div class="stat-box">
                <div>Test Samples</div>
                <div class="stat-val">{{ num_samples }}</div>
            </div>
        </div>

        <div class="plots">
            <div><img src="data:image/png;base64,{{ plot_pnl }}"></div>
            <div><img src="data:image/png;base64,{{ plot_cm }}"></div>
        </div>

        <h3>Recent Trades (Last 20)</h3>
        {{ table_html|safe }}
    </div>
</body>
</html>
"""

# --- Main Route ---
@app.route('/')
def index():
    # 1. Pipeline
    df, err = get_ohlc_data()
    if df is None:
        return f"<h1>Error</h1><p>{err}</p>"
    
    # 2. Preprocess
    raw_ohlc, derivative_ohlc = process_data(df)
    
    # 3. Create Sequences (Points in 12D)
    # Using derivative_ohlc for features as raw prices don't work with distance metrics over time
    X, y, y_raw_change = create_dataset(derivative_ohlc, df['close'].values, seq_length=3)
    
    # 4. Split 70/30
    split_idx = int(len(X) * 0.70)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    y_change_test = y_raw_change[split_idx:]
    
    # 5. Inference (The Painting)
    # Normalize data for distance calculation logic (crude normalization based on train stats)
    # This ensures the "Sphere Radius" is meaningful across different price scales
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0) + 1e-8
    
    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std
    
    preds = infer_painting(X_train_norm, y_train, X_test_norm)
    
    # 6. Calc Metrics
    correct = (preds == y_test)
    accuracy = np.mean(correct) * 100
    
    # PnL: If pred is 1, return is change. If -1, return is -change. If 0, 0.
    # We use y_change_test (the actual % movement)
    strategy_returns = preds * y_change_test
    total_pnl = np.sum(strategy_returns) * 100
    
    # 7. Formatting for Web
    # Create DataFrame for display
    results_df = pd.DataFrame({
        'Actual_Dir': y_test,
        'Predicted_Dir': preds,
        'Actual_Return': y_change_test,
        'Strat_Return': strategy_returns
    })
    
    table_html = results_df.tail(20).to_html(classes='table', float_format=lambda x: '%.5f' % x)
    plot_pnl, plot_cm = create_plots(y_test, preds, strategy_returns)
    
    return render_template_string(HTML_TEMPLATE, 
                                  accuracy=f"{accuracy:.2f}", 
                                  total_pnl=f"{total_pnl:.2f}",
                                  num_samples=len(X_test),
                                  plot_pnl=plot_pnl,
                                  plot_cm=plot_cm,
                                  table_html=table_html)

if __name__ == '__main__':
    print(f"Starting server on port {PORT}...")
    app.run(host='0.0.0.0', port=PORT, debug=False)
