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
from flask import Flask, render_template_string
from sklearn.neighbors import RadiusNeighborsClassifier

# --- Configuration ---
DATA_PATH = '/app/data/ohlc.csv'
MODEL_PATH = '/app/data/painting_model.pkl'
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1h'
LIMIT = 2000
THRESHOLD = 0.001
# Radius acts as the "size" of your fading sphere. 
# Since we normalize data, 2.0 is a generous overlap size (2 standard deviations).
SPHERE_RADIUS = 2.0 
PORT = 8080

app = Flask(__name__)

# --- Custom Fading Sphere Logic ---
# This function defines your exact "painting" math: 
# Intensity = 1 at center, 0 at edge.
def fading_sphere_kernel(distances):
    # distances is an array of distances from the test point to neighbors
    # We clip to ensure no negative values, though radius search prevents this usually
    weights = np.maximum(0, 1 - (distances / SPHERE_RADIUS))
    return weights

# --- 1. Fetch Data ---
def get_ohlc_data():
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    
    print("Fetching data from Binance...")
    exchange = ccxt.binance()
    ohlc = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=LIMIT)
    df = pd.DataFrame(ohlc, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df.to_csv(DATA_PATH, index=False)
    return df

# --- 2. Process & Feature Engineering ---
def prepare_data(df):
    raw_ohlc = df[['open', 'high', 'low', 'close']].values
    # Derivative OHLC (first entry 0)
    derivative = np.diff(raw_ohlc, axis=0, prepend=0)
    
    X = []
    y = []
    y_raw = []
    
    seq_len = 3
    # Create 12D points
    for i in range(len(derivative) - seq_len):
        X.append(derivative[i:i+seq_len].flatten())
        
        # Target: 4th candle direction
        curr = df['close'].values[i+seq_len-1]
        next_val = df['close'].values[i+seq_len]
        change = (next_val - curr) / curr
        y_raw.append(change)
        
        if change > THRESHOLD: y.append(1.0)
        elif change < -THRESHOLD: y.append(-1.0)
        else: y.append(0.0)
            
    return np.array(X), np.array(y), np.array(y_raw)

# --- 3. The Efficient Painting (Training) ---
def train_painting(X_train, y_train):
    print("Painting the canvas (Building Spatial Tree)...")
    
    # We use a RadiusNeighborsClassifier.
    # This spatially indexes the points. It is the "efficient" way to store the painting.
    # weights=fading_sphere_kernel applies your specific 1 -> 0 fading logic.
    clf = RadiusNeighborsClassifier(radius=SPHERE_RADIUS, 
                                    weights=fading_sphere_kernel, 
                                    algorithm='auto', 
                                    outlier_label=0) # If no spheres touch, assume 0 (Flat)
    
    clf.fit(X_train, y_train)
    return clf

# --- 4. Plotting ---
def create_plots(y_true, y_pred, returns):
    # PnL Curve
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(np.cumsum(returns)*100, color='#00ff88', linewidth=2)
    ax1.set_facecolor('#222')
    fig1.patch.set_facecolor('#222')
    ax1.tick_params(colors='white')
    ax1.set_title('Accumulated PnL (%)', color='white')
    ax1.grid(True, alpha=0.1)
    
    buf1 = io.BytesIO()
    fig1.savefig(buf1, format='png', bbox_inches='tight')
    plot_pnl = base64.b64encode(buf1.getvalue()).decode()
    plt.close(fig1)
    
    # Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=[-1, 0, 1])
    fig2, ax2 = plt.subplots(figsize=(5, 5))
    ax2.imshow(cm, cmap='Blues')
    ax2.set_title('Confusion Matrix')
    
    # Annotate
    for i in range(3):
        for j in range(3):
            ax2.text(j, i, cm[i, j], ha='center', va='center', 
                     color='white' if cm[i,j] > cm.max()/2 else 'black')
    
    buf2 = io.BytesIO()
    fig2.savefig(buf2, format='png', bbox_inches='tight')
    plot_cm = base64.b64encode(buf2.getvalue()).decode()
    plt.close(fig2)
    
    return plot_pnl, plot_cm

# --- Main Logic ---
RESULTS = {}

def update_system():
    # 1. Get Data
    df = get_ohlc_data()
    X, y, y_raw = prepare_data(df)
    
    # 2. Split
    split = int(len(X) * 0.7)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    y_raw_test = y_raw[split:]
    
    # 3. Normalize (Crucial for sphere geometry)
    # We save the scaler stats to apply to test data
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    
    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std
    
    # 4. Load or Train Model
    if os.path.exists(MODEL_PATH):
        print("Loading existing painting...")
        clf = joblib.load(MODEL_PATH)
    else:
        clf = train_painting(X_train_norm, y_train)
        print("Saving painting to disk...")
        joblib.dump(clf, MODEL_PATH)
        
    # 5. Inference (Fast Lookup)
    print("Looking up test points in the painting...")
    # This is O(N_test * log(N_train)) - much faster than brute force
    preds = clf.predict(X_test_norm)
    
    # 6. Stats
    acc = np.mean(preds == y_test) * 100
    strat_returns = preds * y_raw_test
    total_pnl = np.sum(strat_returns) * 100
    
    plot_pnl, plot_cm = create_plots(y_test, preds, strat_returns)
    
    RESULTS['acc'] = f"{acc:.2f}"
    RESULTS['pnl'] = f"{total_pnl:.2f}"
    RESULTS['count'] = len(X_test)
    RESULTS['plot_pnl'] = plot_pnl
    RESULTS['plot_cm'] = plot_cm
    
    # Table data
    df_res = pd.DataFrame({
        'Pred': preds,
        'Actual': y_test,
        'Return': y_raw_test
    })
    RESULTS['table'] = df_res.tail(15).to_html(classes='table', index=False)

# --- Routes ---
HTML = """
<!DOCTYPE html>
<body style="background:#111; color:#eee; font-family:sans-serif; padding:2rem">
    <h1 style="color:#0f0">13D Spatial Index Results</h1>
    <div style="background:#222; padding:1rem; border-radius:8px; margin-bottom:1rem">
        Accuracy: <b>{{ r.acc }}%</b> | PnL: <b>{{ r.pnl }}%</b> | Samples: {{ r.count }}
    </div>
    <div style="display:flex; gap:1rem">
        <img src="data:image/png;base64,{{ r.plot_pnl }}" style="border-radius:8px">
        <img src="data:image/png;base64,{{ r.plot_cm }}" style="border-radius:8px">
    </div>
    <h3>Recent Vectors</h3>
    <style>table{width:100%; border-collapse:collapse} td,th{padding:8px; border-bottom:1px solid #444}</style>
    {{ r.table|safe }}
</body>
"""

@app.route('/')
def home():
    if not RESULTS:
        return "Initializing..."
    return render_template_string(HTML, r=RESULTS)

if __name__ == '__main__':
    update_system()
    app.run(host='0.0.0.0', port=PORT)
