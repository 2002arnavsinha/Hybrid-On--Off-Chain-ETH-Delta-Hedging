# volhedge_gp_live.py
# LIVE VolHedge + GP Smile → PerpHedge (Sepolia)
# NO DERIBIT. ONLY CSV + SMART SIMULATION FALLBACK
# Alexander & Imeraj (2023) → On-Chain Delta Hedge

from web3 import Web3
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
import time
import json
import csv
from datetime import datetime
import os

# ==============================================================================
# 1. CONFIG
# ==============================================================================
INFURA_URL = "https://sepolia.infura.io/v3/d4d0d9ec71134849b9ebfdc52f31d9b2"
PRIVATE_KEY = "18b0429ff75b9fa007b9a1e5ea1e66be364b33404dcfcfd8fa57ce8165fd81f8"
PERP_ADDRESS = "0x6c0813E22d8d45748A6cfB18632Dd67855f90D06"
USER_ADDRESS = "0xd64B9edcdFb2871D93cfF7e6C488E05a13ecBc34"

w3 = Web3(Web3.HTTPProvider(INFURA_URL))
account = w3.eth.account.from_key(PRIVATE_KEY)

# Load ABI
with open('/Users/veronica/Desktop/Aryan/Sem 3 /Crypto/Project Crypto/PerpHedge.json') as f:
    ABI = json.load(f)['abi']
contract = w3.eth.contract(address=PERP_ADDRESS, abi=ABI)

LOG_FILE = 'volhedge_gp_live_log.csv'
CSV_PATH = 'data/btc_iv_history.csv'

# ==============================================================================
# GLOBAL STATE (FIXED: moved outside loop)
# ==============================================================================
df_hist = None  # Will be loaded once, then updated daily

# ==============================================================================
# 1.5. LOAD REAL CSV + SIMULATION FALLBACK
# ==============================================================================
def load_btc_data():
    if os.path.exists(CSV_PATH):
        try:
            df = pd.read_csv(CSV_PATH, parse_dates=['Dates'])
            required = ['Dates', 'Close', 'ATM_IV', 'IV_25D_PUT']
            if all(col in df.columns for col in required):
                df = df[required].sort_values('Dates').reset_index(drop=True)
                df = df.tail(90).reset_index(drop=True)
                print(f"Loaded {len(df)} rows from {CSV_PATH}")
                return df
        except Exception as e:
            print(f"CSV load failed: {e}")

    # === SIMULATION FALLBACK ===
    print("CSV not found or invalid → generating realistic simulation...")
    np.random.seed(int(time.time()) % 2**32)
    n_days = 90
    dates = pd.date_range(end=pd.Timestamp.now(), periods=n_days, freq='D')

    trend = np.linspace(60000, 68000, n_days)
    noise = np.cumsum(np.random.randn(n_days) * 800)
    prices = trend + noise
    prices = np.clip(prices, 30000, 100000)

    base_atm = 65.0
    atm_iv = base_atm + np.random.randn(n_days) * 15
    atm_iv = np.clip(atm_iv, 40, 120)

    skew = 15 + 20 * (prices[-1] - prices) / (prices.max() - prices.min())
    otm_iv = atm_iv + skew + np.random.randn(n_days) * 5
    otm_iv = np.clip(otm_iv, atm_iv + 5, 150)

    df_sim = pd.DataFrame({
        'Dates': dates.date,
        'Close': prices,
        'ATM_IV': atm_iv,
        'IV_25D_PUT': otm_iv
    })

    os.makedirs('data', exist_ok=True)
    df_sim.to_csv(CSV_PATH, index=False)
    print(f"Simulation saved to {CSV_PATH}")
    return df_sim.tail(90).reset_index(drop=True)

# ==============================================================================
# 2. BLACK-SCHOLES GREEKS
# ==============================================================================
def bs_delta_usd(F, K, T, sigma, r=0.0, option_type='put'):
    if T <= 0 or sigma <= 0:
        return -1.0 if (option_type == 'put' and F < K) else 0.0
    d1 = (np.log(F / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return -norm.cdf(-d1) if option_type == 'put' else norm.cdf(d1)

def bs_vega_usd(F, K, T, sigma, r=0.0):
    if T <= 0 or sigma <= 0: return 0.0
    d1 = (np.log(F / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return F * np.sqrt(T) * norm.pdf(d1) / 100.0

# ==============================================================================
# 3. GAUSSIAN PROCESS SMILE SLOPE
# ==============================================================================
def rbf_kernel(X1, X2, l=1.0, sigma_f=1.0):
    X1 = np.atleast_2d(X1)
    X2 = np.atleast_2d(X2)
    sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
    return sigma_f**2 * np.exp(-0.5 / l**2 * sqdist)

def neg_log_marginal_lik(params, X, y):
    log_l, log_sigma_f, log_sigma_n = params
    l = np.exp(log_l)
    sigma_f = np.exp(log_sigma_f)
    sigma_n = np.exp(log_sigma_n)
    n = len(X)
    K = rbf_kernel(X, X, l, sigma_f) + sigma_n**2 * np.eye(n)
    try:
        L = np.linalg.cholesky(K + 1e-6 * np.eye(n))
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
        log_lik = -0.5 * np.dot(y.T, alpha) - np.sum(np.log(np.diag(L))) - 0.5 * n * np.log(2 * np.pi)
        return -log_lik
    except np.linalg.LinAlgError:
        return np.inf

def gp_fit_predict(X, y, X_star, l=0.5, sigma_f=0.05, sigma_n=0.01):
    X = np.atleast_2d(X).T if X.ndim == 1 else X
    X_star = np.atleast_2d(X_star).T if X_star.ndim == 1 else X_star
    n = X.shape[0]
    K = rbf_kernel(X, X, l, sigma_f) + sigma_n**2 * np.eye(n)
    try:
        L = np.linalg.cholesky(K + 1e-6 * np.eye(n))
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
        K_star = rbf_kernel(X, X_star, l, sigma_f)
        mean = np.dot(K_star.T, alpha)
        v = np.linalg.solve(L, K_star)
        var = rbf_kernel(X_star, X_star, l, sigma_f) - np.dot(v.T, v)
        return mean.flatten(), np.diag(var)
    except np.linalg.LinAlgError:
        return np.zeros(len(X_star)), np.zeros(len(X_star))

def compute_smile_slope(df_window, S_current, K_fixed):
    if len(df_window) < 10: return 0.0, 0.0
    S = df_window['Close'].values
    m = K_fixed / S
    log_m = np.log(m)
    iv_atm = df_window['ATM_IV'].values / 100.0
    iv_otm = df_window['IV_25D_PUT'].values / 100.0
    mask = (m >= 0.6) & (m <= 1.4)
    if mask.sum() < 5: return 0.0, 0.0
    X = log_m[mask].reshape(-1, 1)
    y = iv_otm[mask] - iv_atm[mask]
    if np.std(X) < 1e-6 or np.std(y) < 1e-6: return 0.0, 0.0

    initial_params = np.log([0.5, 0.05, 0.01])
    bounds = [(-2, 2), (-5, 0), (-10, -1)]
    res = minimize(neg_log_marginal_lik, initial_params, args=(X, y), method='L-BFGS-B', bounds=bounds, options={'maxiter': 100})
    l, sigma_f, sigma_n = np.exp(res.x) if res.success else (0.5, 0.05, 0.01)

    m_now = K_fixed / S_current
    log_m_now = np.log(m_now)
    eps = 0.01
    X_left = np.array([log_m_now - eps])
    X_right = np.array([log_m_now + eps])
    X_center = np.array([log_m_now])

    mean_center, var_center = gp_fit_predict(X, y, X_center, l, sigma_f, sigma_n)
    mean_left, var_left = gp_fit_predict(X, y, X_left, l, sigma_f, sigma_n)
    mean_right, var_right = gp_fit_predict(X, y, X_right, l, sigma_f, sigma_n)

    slope_mean = (mean_right[0] - mean_left[0]) / (2 * eps)
    slope_std = np.sqrt(var_left[0] + var_right[0]) / (2 * eps)
    dsigma_dm_mean = slope_mean / m_now if abs(m_now) > 1e-6 else 0.0
    dsigma_dm_std = slope_std / m_now if abs(m_now) > 1e-6 else 0.0

    if np.isnan(dsigma_dm_mean) or abs(m_now - 1.0) < 1e-2:
        dsigma_dm_mean = mean_center[0] / (m_now - 1.0) if abs(m_now - 1.0) > 1e-6 else slope_mean
        dsigma_dm_std = np.sqrt(var_center[0]) / abs(m_now - 1.0) if abs(m_now - 1.0) > 1e-6 else slope_std

    return dsigma_dm_mean, dsigma_dm_std

# ==============================================================================
# 4. SM DELTA WITH 95% CI
# ==============================================================================
def sm_delta_usd(F, K, T, sigma, smile_slope_mean, smile_slope_std, alpha=1.96):
    smile_slope = smile_slope_mean + alpha * smile_slope_std
    delta_bs = bs_delta_usd(F, K, T, sigma, option_type='put')
    vega_norm = bs_vega_usd(F, K, T, sigma) / F
    m = K / F
    return np.clip(delta_bs + vega_norm * smile_slope * (-m), -1.0, 0.0)

# volhedge_gp_live.py
# ==============================================================================
# 5. SEND TX (EIP-1559 + AUTO NONCE FIX)
# ==============================================================================
def send_tx(func, *args):
    # Get current nonce
    nonce = w3.eth.get_transaction_count(account.address)
    
    tx = func(*args).build_transaction({
        'from': account.address,
        'nonce': nonce,
        'gas': 400000,
        'maxFeePerGas': w3.to_wei('2.5', 'gwei'),
        'maxPriorityFeePerGas': w3.to_wei('0.1', 'gwei'),
        'chainId': 11155111
    })
    signed = account.sign_transaction(tx)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    print(f"Tx sent: {tx_hash.hex()}")
    
    try:
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=180)
        print(f"Tx confirmed in block {receipt.blockNumber}")
    except Exception as e:
        print(f"Tx failed: {e}")

# ==============================================================================
# 6. LIVE LOOP
# ==============================================================================
df_hist = None

print("GP VolHedge Bot Starting... (CSV + Simulation Fallback)")

while True:
    try:
        print(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M')} | GP VolHedge Live Cycle")

        # 1. Get BTC price
        btc_price = contract.functions.getBTCPricePublic().call() / 1e8
        print(f"BTC Price: ${btc_price:,.2f}")

        # 2. Load or update historical data
        if df_hist is None:
            df_hist = load_btc_data()

        # 3. Append today
        latest_atm = df_hist['ATM_IV'].iloc[-1]
        latest_otm = df_hist['IV_25D_PUT'].iloc[-1]

        today_df = pd.DataFrame([{
            'Dates': pd.Timestamp.now().date(),
            'Close': btc_price,
            'ATM_IV': latest_atm,
            'IV_25D_PUT': latest_otm
        }])
        df_hist = pd.concat([df_hist, today_df], ignore_index=True).tail(90)

        # 4. Option params
        S0 = df_hist['Close'].iloc[-2]
        K = 1.25 * S0
        T = 21 / 365.0
        sigma = df_hist['ATM_IV'].iloc[-1] / 100.0

        # 5. GP Smile Slope
        window = df_hist.iloc[-60:-1]
        slope_mean, slope_std = compute_smile_slope(window, btc_price, K)
        print(f"GP Smile Slope: {slope_mean:.6f} ± {slope_std:.6f}")

        # 6. SM Delta (95% CI)
        delta = sm_delta_usd(btc_price, K, T, sigma, slope_mean, slope_std, alpha=1.96)
        ratio = int(abs(delta) * 10000)
        print(f"SM Delta (95% CI): {delta:.6f} → HedgeRatio: {ratio}")

        # 7. CHECK POSITION & UPDATE
        position = contract.functions.getPositionPacked(USER_ADDRESS).call()
        if position[0] == 0:
            print("No open position. Skipping updates.")
        else:
            send_tx(contract.functions.updateHedgeRatio, ratio)
            send_tx(contract.functions.updateVolatility, int(sigma * 10000))

        # 8. Log PnL
        pnl = contract.functions.getPnL(USER_ADDRESS).call() / 1e18
        print(f"On-chain PnL: {pnl:+.6f} HEDGE")

        # 9. CSV LOG
        with open(LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(['timestamp', 'btc_price', 'slope_mean', 'slope_std', 'sm_delta', 'hedge_ratio', 'volatility', 'pnl'])
            writer.writerow([datetime.now(), btc_price, slope_mean, slope_std, delta, ratio, int(sigma * 10000), pnl])

        print("Cycle complete. Sleeping 24h...")
        time.sleep(10)  # 60 for testing

    except Exception as e:
        print("ERROR:", e)
        time.sleep(300)