import os
import ccxt
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from flask import Flask, render_template_string

# --- Configuration ---
DATA_PATH = '/app/data/ohlc.csv'
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1h'
LIMIT = 2000
THRESHOLD = 0.001
SPHERE_RADIUS = 0.5  # Adjusted for normalized data
PORT = 8080

# --- Global Storage ---
# We store the results here so the web page loads instantly
RESULTS_CACHE = {
    'accuracy': None,
    'total_pnl': None,
    'num_samples': 0,
    'plot_pnl': None,
    'plot_cm': None,
    'table_html': None
}

app = Flask(__name__)

# --- 1. Fetch & 2/3. Save/Load ---
def get_ohlc_data():
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    if os.path.exists(DATA_PATH):
        print(f"Loading data from {DATA_PATH}...")
        return pd.read_csv(DATA_PATH)
    
    print(f"Fetching {LIMIT} candles for {SYMBOL}...")
    exchange = ccxt.binance()
    ohlc = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=LIMIT)
    df = pd.DataFrame(ohlc, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df.to_csv(DATA_PATH, index=False)
    return df

# --- 4. Processing ---
def process_data(df):
    raw_ohlc = df[['open', 'high', 'low', 'close']].values
    # Derivative: Current - Previous. Prepend 0 to match size.
    derivative_ohlc = np.diff(raw_ohlc, axis=0, prepend=0)
    return raw_ohlc, derivative_ohlc

# --- 6. Sequences & 7. Labeling ---
def create_dataset(data, raw_close, seq_length=3):
    X = []
    y = []
    y_raw_change = []

    for i in range(len(data) - seq_length):
        # 12 Dimensions (3 candles * 4 features)
        seq = data[i:i+seq_length].flatten()
        X.append(seq)
        
        # Label logic based on 4th candle
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
    print("Building 13D Painting and running inference...")
    
    # 13D Training Points: 12 features + 1 label dimension
    train_points_13d = np.hstack((X_train, y_train.reshape(-1, 1)))
    
    predictions = []
    candidates = [-1.0, 0.0, 1.0]
    
    # Vectorized inference could be optimized further, but sticking to loop for clarity of "Sphere" logic
    for i, x_t in enumerate(X_test):
        best_intensity = -1.0
        best_dir = 0.0
        
        for cand_dir in candidates:
            # Hypothetical 13D point
            test_point_13d = np.append(x_t, cand_dir)
            
            # Distance to all training spheres
            dists = np.linalg.norm(train_points_13d - test_point_13d, axis=1)
            
            # Fading sphere: Intensity = 1 at center, 0 at radius
            intensities = np.maximum(0, 1 - (dists / SPHERE_RADIUS))
            
            # Sum of paint intensity for this hypothesis
            total_intensity = np.sum(intensities)
            
            if total_intensity > best_intensity:
                best_intensity = total_intensity
                best_dir = cand_dir
        
        predictions.append(best_dir)
        
        if i % 100 == 0:
            print(f"Processed {i}/{len(X_test)} test points...")
            
    return np.array(predictions)

# --- Plotting ---
def generate_assets(y_test, preds, strategy_returns):
    # PnL Plot
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    cumulative = np.cumsum(strategy_returns) * 100
    ax1.plot(cumulative, label='Cumulative PnL (%)', color='#00ff88')
    ax1.set_facecolor('#1e1e1e')
    fig1.patch.set_facecolor('#1e1e1e')
    ax1.tick_params(colors='white')
    ax1.xaxis.label.set_color('white')
    ax1.yaxis.label.set_color('white')
    ax1.set_title('Strategy Performance', color='white')
    ax1.grid(True, alpha=0.1)
    
    img1 = io.BytesIO()
    fig1.savefig(img1, format='png', bbox_inches='tight')
    img1.seek(0)
    plot_pnl = base64.b64encode(img1.getvalue()).decode()
    plt.close(fig1)
    
    # Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, preds, labels=[-1, 0, 1])
    
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    im = ax2.imshow(cm, interpolation='nearest', cmap='Blues')
    ax2.set_title('Confusion Matrix')
    # Label axes
    ax2.set_xticks([0, 1, 2])
    ax2.set_yticks([0, 1, 2])
    ax2.set_xticklabels(['Short', 'Flat', 'Long'])
    ax2.set_yticklabels(['Short', 'Flat', 'Long'])
    
    # Add text annotations
    for i in range(3):
        for j in range(3):
            ax2.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > cm.max()/2 else "black")
    
    img2 = io.BytesIO()
    fig2.savefig(img2, format='png', bbox_inches='tight')
    img2.seek(0)
    plot_cm = base64.b64encode(img2.getvalue()).decode()
    plt.close(fig2)
    
    return plot_pnl, plot_cm

# --- Initialization Pipeline ---
def run_pipeline():
    df = get_ohlc_data()
    raw, derivative = process_data(df)
    
    # Create sequences (3 candles = 12 dims)
    X, y, y_raw_change = create_dataset(derivative, df['close'].values, seq_length=3)
    
    # Split 70/30
    split = int(len(X) * 0.7)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    y_change_test = y_raw_change[split:]
    
    # Normalization (Crucial for Euclidean distance)
    # We normalize based on Train stats only
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0) + 1e-8
    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std
    
    # Inference
    preds = infer_painting(X_train_norm, y_train, X_test_norm)
    
    # Metrics
    accuracy = np.mean(preds == y_test) * 100
    strategy_returns = preds * y_change_test
    total_pnl = np.sum(strategy_returns) * 100
    
    # Store results
    plot_pnl, plot_cm = generate_assets(y_test, preds, strategy_returns)
    
    res_df = pd.DataFrame({
        'Predicted': preds,
        'Actual': y_test,
        'Return': y_change_test,
        'Strat_PnL': strategy_returns
    })
    
    RESULTS_CACHE['accuracy'] = f"{accuracy:.2f}"
    RESULTS_CACHE['total_pnl'] = f"{total_pnl:.2f}"
    RESULTS_CACHE['num_samples'] = len(X_test)
    RESULTS_CACHE['plot_pnl'] = plot_pnl
    RESULTS_CACHE['plot_cm'] = plot_cm
    RESULTS_CACHE['table_html'] = res_df.tail(20).to_html(classes='table', index=False)
    
    print("Pipeline complete. Server ready.")

# --- Web Server ---
HTML = """
<!DOCTYPE html>
<head>
<style>
    body { font-family: monospace; background: #121212; color: #e0e0e0; padding: 20px; }
    .card { background: #1e1e1e; padding: 20px; margin-bottom: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
    h1 { color: #00ff88; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
    table { width: 100%; border-collapse: collapse; }
    th { text-align: left; border-bottom: 1px solid #333; color: #888; }
    td { padding: 8px 0; border-bottom: 1px solid #222; }
    img { max-width: 100%; border-radius: 4px; }
</style>
</head>
<body>
    <h1>13-Dimensional Sphere Backtest</h1>
    <div class="card">
        <h2>Metrics</h2>
        <p>Accuracy: <b>{{ r.accuracy }}%</b> | Total PnL: <b>{{ r.total_pnl }}%</b> | Samples: {{ r.num_samples }}</p>
    </div>
    <div class="grid">
        <div class="card"><img src="data:image/png;base64,{{ r.plot_pnl }}"></div>
        <div class="card"><img src="data:image/png;base64,{{ r.plot_cm }}"></div>
    </div>
    <div class="card">
        <h3>Recent Classification Logs</h3>
        {{ r.table_html|safe }}
    </div>
</body>
"""

@app.route('/')
def index():
    if RESULTS_CACHE['accuracy'] is None:
        return "System initializing... please refresh in a moment."
    return render_template_string(HTML, r=RESULTS_CACHE)

if __name__ == '__main__':
    # Run the heavy pipeline once at startup
    run_pipeline()
    app.run(host='0.0.0.0', port=PORT)
