import os
import ccxt
import pandas as pd
import numpy as np
import time
from datetime import datetime

# --- Configuration for Grid Search ---
TIMEFRAME = os.environ.get('TIMEFRAME', '1h')
SYMBOL = os.environ.get('SYMBOL', 'BTC/USDT')
START = os.environ.get('START', '2024-01-01 00:00:00')
END = os.environ.get('END', '2024-06-01 00:00:00')
SPLIT_RATIO = 0.7  # 70% Training, 30% Testing

# Grid Search Space
C_VALUES = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32]
D_VALUES = [4, 5]
E_VALUES = [0.001, 0.002, 0.004, 0.008, 0.016]

# --- Core Functions ---

def fetch(timeframe, symbol, start_str, end_str):
    """Fetches OHLCV data from Binance."""
    print(f"Fetching {symbol} {timeframe} from {start_str} to {end_str}...")
    exchange = ccxt.binance()
    start_ts = exchange.parse8601(start_str)
    end_ts = exchange.parse8601(end_str)
    
    ohlc = []
    current_ts = start_ts
    
    while current_ts < end_ts:
        try:
            candles = exchange.fetch_ohlcv(symbol, timeframe, since=current_ts, limit=1000)
            if not candles:
                break
            candles = [c for c in candles if c[0] < end_ts]
            if not candles:
                break
            ohlc += candles
            current_ts = candles[-1][0] + 1
            time.sleep(0.05) 
        except Exception as e:
            print(f"Error fetching: {e}")
            break
            
    df = pd.DataFrame(ohlc, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    cols = ['open', 'high', 'low', 'close', 'volume']
    df[cols] = df[cols].astype(float)
    return df

def deriveround(df):
    """Applies returns calculation."""
    df = df.copy()
    cols = ['open', 'high', 'low', 'close']
    for col in cols:
        df[f'{col}_ret'] = df[col].pct_change().fillna(0.0)
    return df

def split(df, b):
    split_idx = int(len(df) * b)
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()

def get_windows(df, d):
    """Extracts windows from dataframe without calculating density."""
    data_cols = ['open_ret', 'high_ret', 'low_ret', 'close_ret']
    data_values = df[data_cols].values.copy()
    
    window_list = []
    num_windows = len(data_values) - d + 1
    
    if num_windows < 1:
        return np.array([])

    for i in range(num_windows):
        window = data_values[i : i+d]
        window_list.append(window)
        
    return np.array(window_list)

def compute_densities_and_sort(windows, e):
    """Calculates density for windows and returns them sorted by density."""
    N = len(windows)
    if N == 0:
        return np.array([])
        
    flat_windows = windows.reshape(N, -1)
    densities = np.zeros(N, dtype=int)
    chunk_size = 1000
    
    # Calculate density
    for i in range(0, N, chunk_size):
        end = min(i + chunk_size, N)
        batch = flat_windows[i:end]
        compare_set = flat_windows if N < 10000 else flat_windows[::5]
        
        for j in range(len(batch)):
            diff = np.abs(compare_set - batch[j])
            matches = np.all(diff < e, axis=1)
            densities[i+j] = np.sum(matches)
    
    # Sort indices by density (ascending)
    sorted_indices = np.argsort(densities)
    return windows[sorted_indices] # Returns windows sorted from low density to high density

def backtest(df_target, model_patterns, e, d):
    """Runs the prediction loop on the test set."""
    if len(df_target) < d:
        return 0, 0, 0.0

    # 1. Prepare Target Windows
    ret_cols = ['open_ret', 'high_ret', 'low_ret', 'close_ret']
    ret_values = df_target[ret_cols].values.copy()
    
    window_list = []
    num_windows = len(ret_values) - d + 1
    
    for i in range(num_windows):
        window_list.append(ret_values[i : i+d])
        
    ret_windows = np.array(window_list) 
    
    # Model Setup
    model_context = model_patterns[:, :d-1, :]
    model_outcome = model_patterns[:, -1, :]
    model_context_flat = model_context.reshape(model_context.shape[0], -1)
    
    total_trades = 0
    correct_trades = 0
    cumulative_pnl = 0.0
    
    # 2. Iterate
    for i in range(len(ret_windows)):
        current_context_ret = ret_windows[i, :d-1, :]
        current_context_flat = current_context_ret.reshape(-1)
        
        diff = np.abs(model_context_flat - current_context_flat)
        matches_idx = np.where(np.all(diff < e, axis=1))[0]
        
        if len(matches_idx) > 0:
            matched_outcomes = model_outcome[matches_idx]
            avg_return = np.mean(matched_outcomes[:, 3])
            
            predicted_dir = 1 if avg_return > 0 else -1
            if avg_return == 0: predicted_dir = 0
            
            if predicted_dir != 0:
                target_idx = i + d - 1
                actual_ret = df_target.iloc[target_idx]['close_ret']
                actual_dir = 1 if actual_ret > 0 else -1
                if actual_ret == 0: actual_dir = 0
                
                is_correct = (predicted_dir == actual_dir)
                pnl = predicted_dir * actual_ret
                
                total_trades += 1
                if is_correct:
                    correct_trades += 1
                cumulative_pnl += pnl

    return total_trades, correct_trades, cumulative_pnl

# --- Main Grid Search ---

def run_grid_search():
    # 1. Prepare Data
    df = fetch(TIMEFRAME, SYMBOL, START, END)
    if len(df) == 0:
        print("No data fetched. Exiting.")
        return

    df = deriveround(df)
    train_df, test_df = split(df, SPLIT_RATIO)
    
    print(f"Data Loaded: {len(df)} rows. Train: {len(train_df)}, Test: {len(test_df)}")
    print(f"Starting Grid Search...")
    print(f"Param Space: C={C_VALUES}, D={D_VALUES}, E={E_VALUES}")
    print("-" * 80)
    print(f"{'D':<3} | {'E':<6} | {'C':<5} | {'Trades':<6} | {'Acc %':<8} | {'PnL':<8} | {'Score'}")
    print("-" * 80)

    results = []

    # Outer Loop: D (Sequence Length)
    # Changing D changes the shape of the data, so we re-extract windows here.
    for d in D_VALUES:
        train_windows = get_windows(train_df, d)
        
        # Middle Loop: E (Similarity Threshold)
        # Changing E changes the density calculation (what counts as a neighbor).
        for e in E_VALUES:
            # Expensive Step: Calculate density and sort training windows
            # We do this once per (D, E) pair, then slice for different C values.
            sorted_windows = compute_densities_and_sort(train_windows, e)
            total_windows = len(sorted_windows)
            
            # Inner Loop: C (Top % Density)
            # Just slicing the pre-sorted windows. Fast.
            for c in C_VALUES:
                top_n = int(total_windows * c)
                if top_n == 0: top_n = 1
                
                # Take the top N densest windows (end of the sorted array)
                model_patterns = sorted_windows[-top_n:]
                
                # Backtest
                trades, correct, pnl = backtest(test_df, model_patterns, e, d)
                
                accuracy = (correct / trades * 100) if trades > 0 else 0.0
                
                # Simple score to balance activity and accuracy
                # Score = PnL, but heavily penalized if trades are too low (< 5)
                score = pnl if trades > 5 else -100
                
                results.append({
                    'D': d, 'E': e, 'C': c,
                    'Trades': trades, 'Acc': accuracy, 'PnL': pnl, 'Score': score
                })
                
                print(f"{d:<3} | {e:<6} | {c:<5} | {trades:<6} | {accuracy:05.2f}%   | {pnl:06.4f}   | {score:.4f}")

    print("-" * 80)
    print("Top 10 Configurations by PnL:")
    
    # Sort by PnL Descending
    results.sort(key=lambda x: x['PnL'], reverse=True)
    
    for i, r in enumerate(results[:10]):
        print(f"{i+1}. D={r['D']}, E={r['E']}, C={r['C']} -> PnL: {r['PnL']:.4f}, Acc: {r['Acc']:.2f}%, Trades: {r['Trades']}")

if __name__ == "__main__":
    run_grid_search()
