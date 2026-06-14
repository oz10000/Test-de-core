#!/usr/bin/env python3
# run_all_research_tests.py
# Empirical audit of signal_engine.py for PyDROID SOLANA CORE Ω
# Runs on Python 3.11+ (compatible with Pydroid).
# Outputs to console (flush) and persistent log file.
# Output directory configurable via OUTPUT_DIR env var.

import os
import sys
import time
import math
import random
import json
import logging
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import requests

# Optional plotting
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False

# ==================== CONFIGURATION ====================
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./reports")
os.makedirs(OUTPUT_DIR, exist_ok=True)

LOG_FILE = os.path.join(OUTPUT_DIR, "audit.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("ResearchAudit")

SYMBOL_SOL = "SOLUSDT"
SYMBOL_BTC = "BTCUSDT"
SYMBOL_ETH = "ETHUSDT"
TIMEFRAME = "4h"
TARGET_CANDLES = 3000

BINANCE_BASE_URL = "https://api.binance.com/api/v3/klines"
USER_AGENT = "Mozilla/5.0"

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Default keys (will be overridden if signal_engine is available)
SCORE_KEY = "score"
SIGNAL_KEY = "signal"
MACRO_KEY = "macro"

# Flag to indicate if signal_engine is available
SIGNAL_ENGINE_AVAILABLE = False
compute_signal = None

# Try to import signal_engine
try:
    from signal_engine import compute_signal as _compute_signal
    compute_signal = _compute_signal
    SIGNAL_ENGINE_AVAILABLE = True
    logger.info("signal_engine module loaded successfully.")
except ImportError as e:
    logger.error(f"signal_engine not found: {e}. All tests will be skipped.")
except Exception as e:
    logger.error(f"Error loading signal_engine: {e}. All tests will be skipped.")

# ==================== HELPER: INTROSPECT SIGNAL ENGINE ====================
def introspect_signal_engine():
    global SCORE_KEY, SIGNAL_KEY, MACRO_KEY
    if not SIGNAL_ENGINE_AVAILABLE or compute_signal is None:
        return None, set()
    try:
        n = 200
        df = pd.DataFrame({
            "timestamp": pd.date_range("2023-01-01", periods=n, freq="4h"),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1000.0
        })
        res = compute_signal(df, df, df)
        keys = set(res.keys())
        logger.info(f"signal_engine keys detected: {keys}")
        if "score" in keys:
            SCORE_KEY = "score"
        if "signal" in keys:
            SIGNAL_KEY = "signal"
        if "macro" in keys:
            MACRO_KEY = "macro"
        return compute_signal, keys
    except Exception as e:
        logger.error(f"Introspection failed: {e}")
        return None, set()

# ==================== DATA DOWNLOAD (PAGINATED) ====================
def fetch_klines_paginated(symbol: str, interval: str, target_candles: int) -> Optional[pd.DataFrame]:
    limit = 1000
    all_data = []
    start_time = None
    while len(all_data) < target_candles:
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        if start_time:
            params["startTime"] = start_time
        try:
            resp = requests.get(BINANCE_BASE_URL, params=params, timeout=15, headers={"User-Agent": USER_AGENT})
            resp.raise_for_status()
            data = resp.json()
            if not data:
                break
            all_data.extend(data)
            last_ts = data[-1][6]
            start_time = last_ts + 1
        except Exception as e:
            logger.warning(f"Pagination failed for {symbol}: {e}")
            break
    if not all_data:
        return None
    df = pd.DataFrame(all_data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore"
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return df

def load_real_data():
    logger.info("Downloading real data from Binance (paginated)...")
    df_sol = fetch_klines_paginated(SYMBOL_SOL, TIMEFRAME, TARGET_CANDLES)
    df_btc = fetch_klines_paginated(SYMBOL_BTC, TIMEFRAME, TARGET_CANDLES)
    df_eth = fetch_klines_paginated(SYMBOL_ETH, TIMEFRAME, TARGET_CANDLES)
    if df_sol is None or df_btc is None or df_eth is None:
        logger.warning("Real data incomplete, falling back to synthetic.")
        return generate_synthetic_data()
    min_len = min(len(df_sol), len(df_btc), len(df_eth))
    df_sol = df_sol.iloc[:min_len].reset_index(drop=True)
    df_btc = df_btc.iloc[:min_len].reset_index(drop=True)
    df_eth = df_eth.iloc[:min_len].reset_index(drop=True)
    logger.info(f"Real data loaded: {len(df_sol)} candles each.")
    return df_sol, df_btc, df_eth

def generate_synthetic_data():
    n = 3000
    t = np.arange(n)
    corr_matrix = np.array([[1.0, 0.7, 0.6],
                            [0.7, 1.0, 0.8],
                            [0.6, 0.8, 1.0]])
    L = np.linalg.cholesky(corr_matrix)
    uncorrelated = np.random.normal(0, 1, (3, n))
    correlated = L @ uncorrelated
    mu = 0.0001
    sigma = 0.02
    increments = mu + sigma * correlated
    price_sol = 100 * np.exp(np.cumsum(increments[0]))
    price_btc = 30000 * np.exp(np.cumsum(increments[1]))
    price_eth = 2000 * np.exp(np.cumsum(increments[2]))
    price_sol = np.maximum(price_sol, 10)
    price_btc = np.maximum(price_btc, 1000)
    price_eth = np.maximum(price_eth, 100)

    timestamps = pd.date_range("2023-01-01", periods=n, freq="4h")
    df_sol = pd.DataFrame({
        "timestamp": timestamps,
        "open": price_sol,
        "high": price_sol * (1 + 0.01 * np.abs(np.random.randn(n))),
        "low": price_sol * (1 - 0.01 * np.abs(np.random.randn(n))),
        "close": price_sol,
        "volume": np.random.lognormal(15, 1, n)
    })
    df_btc = pd.DataFrame({
        "timestamp": timestamps,
        "open": price_btc,
        "high": price_btc * (1 + 0.005 * np.abs(np.random.randn(n))),
        "low": price_btc * (1 - 0.005 * np.abs(np.random.randn(n))),
        "close": price_btc,
        "volume": np.random.lognormal(20, 1, n)
    })
    df_eth = pd.DataFrame({
        "timestamp": timestamps,
        "open": price_eth,
        "high": price_eth * (1 + 0.007 * np.abs(np.random.randn(n))),
        "low": price_eth * (1 - 0.007 * np.abs(np.random.randn(n))),
        "close": price_eth,
        "volume": np.random.lognormal(18, 1, n)
    })
    return df_sol, df_btc, df_eth

# ==================== TEST FUNCTIONS (all skip if signal_engine unavailable) ====================
def test_integrity(df_sol, df_btc, df_eth):
    logger.info("[INTEGRITY] Starting...")
    issues = []
    for name, df in [("SOL", df_sol), ("BTC", df_btc), ("ETH", df_eth)]:
        required = ["timestamp", "open", "high", "low", "close", "volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            issues.append(f"{name}: missing columns {missing}")
        if df.isnull().any().any():
            issues.append(f"{name}: contains NaN values")
        if len(df) < 100:
            issues.append(f"{name}: insufficient length ({len(df)})")
        if not df["timestamp"].is_monotonic_increasing:
            issues.append(f"{name}: timestamps not sorted")
    passed = len(issues) == 0
    if passed:
        logger.info("[INTEGRITY] OK")
    else:
        logger.warning(f"[INTEGRITY] Issues: {issues}")
    return {"issues": issues, "passed": passed}

def test_mathematical_stability(compute_fn, df_sol, df_btc, df_eth, keys):
    logger.info("[STABILITY] Starting...")
    errors = []
    scores = []
    macro_vals = []
    signals = []
    min_len = min(len(df_sol), len(df_btc), len(df_eth))
    total = min_len - 100
    for i, idx in enumerate(range(100, min_len), start=1):
        if i % 200 == 0:
            logger.info(f"[STABILITY] {i}/{total} candles processed")
        try:
            res = compute_fn(df_sol.iloc[:idx], df_btc.iloc[:idx], df_eth.iloc[:idx])
            score = res.get(SCORE_KEY, 0.0)
            if not isinstance(score, (float, int)) or math.isnan(score) or math.isinf(score):
                errors.append(f"Index {idx}: invalid score {score}")
            scores.append(score)
            if SIGNAL_KEY in res:
                signals.append(res[SIGNAL_KEY])
            if MACRO_KEY in res:
                macro_vals.append(res[MACRO_KEY])
        except Exception as e:
            errors.append(f"Index {idx}: exception {e}")
    logger.info(f"[STABILITY] Completed: {len(scores)} valid scores, {len(errors)} errors")
    return {"scores": scores, "macro_vals": macro_vals, "signals": signals, "errors": errors}

def test_atr_zero(compute_fn):
    logger.info("[ATR_ZERO] Starting...")
    n = 500
    price = np.ones(n) * 100.0
    df = pd.DataFrame({
        "timestamp": pd.date_range("2023-01-01", periods=n, freq="4h"),
        "close": price,
        "high": price + 0.001,
        "low": price - 0.001,
        "open": price,
        "volume": 1000.0
    })
    scores = []
    errors = []
    for i in range(50, n):
        try:
            res = compute_fn(df.iloc[:i], df.iloc[:i], df.iloc[:i])
            s = res.get(SCORE_KEY, 0.0)
            if not math.isnan(s) and not math.isinf(s):
                scores.append(s)
        except Exception as e:
            errors.append(str(e))
    passed = len(errors) == 0 and len(scores) == n-50
    logger.info(f"[ATR_ZERO] Passed: {passed}")
    return {"passed": passed, "errors": errors}

def test_extreme_correlation(compute_fn):
    logger.info("[CORRELATION] Starting...")
    n = 500
    t = np.arange(n)
    sol = 100 + 0.1 * t + 0.5 * np.random.randn(n)
    df_sol = pd.DataFrame({"close": sol, "high": sol*1.01, "low": sol*0.99, "timestamp": pd.date_range("2023-01-01", periods=n, freq="4h")})
    # positive
    btc_pos = 30000 + 30 * t + 15 * np.random.randn(n)
    eth_pos = 2000 + 2 * t + 1 * np.random.randn(n)
    df_btc = pd.DataFrame({"close": btc_pos, "high": btc_pos*1.01, "low": btc_pos*0.99, "timestamp": df_sol["timestamp"]})
    df_eth = pd.DataFrame({"close": eth_pos, "high": eth_pos*1.01, "low": eth_pos*0.99, "timestamp": df_sol["timestamp"]})
    macro_pos = []
    for i in range(100, n):
        res = compute_fn(df_sol.iloc[:i], df_btc.iloc[:i], df_eth.iloc[:i])
        macro_pos.append(res.get(MACRO_KEY, 0.5))
    # negative
    btc_neg = 30000 - 30 * t + 15 * np.random.randn(n)
    eth_neg = 2000 - 2 * t + 1 * np.random.randn(n)
    df_btc_neg = pd.DataFrame({"close": btc_neg, "high": btc_neg*1.01, "low": btc_neg*0.99, "timestamp": df_sol["timestamp"]})
    df_eth_neg = pd.DataFrame({"close": eth_neg, "high": eth_neg*1.01, "low": eth_neg*0.99, "timestamp": df_sol["timestamp"]})
    macro_neg = []
    for i in range(100, n):
        res = compute_fn(df_sol.iloc[:i], df_btc_neg.iloc[:i], df_eth_neg.iloc[:i])
        macro_neg.append(res.get(MACRO_KEY, 0.5))
    # zero correlation
    btc_rand = 30000 + 100 * np.random.randn(n)
    eth_rand = 2000 + 10 * np.random.randn(n)
    df_btc_rand = pd.DataFrame({"close": btc_rand, "high": btc_rand*1.01, "low": btc_rand*0.99, "timestamp": df_sol["timestamp"]})
    df_eth_rand = pd.DataFrame({"close": eth_rand, "high": eth_rand*1.01, "low": eth_rand*0.99, "timestamp": df_sol["timestamp"]})
    macro_rand = []
    for i in range(100, n):
        res = compute_fn(df_sol.iloc[:i], df_btc_rand.iloc[:i], df_eth_rand.iloc[:i])
        macro_rand.append(res.get(MACRO_KEY, 0.5))
    result = {
        "macro_pos_mean": float(np.mean(macro_pos)) if macro_pos else 0,
        "macro_neg_mean": float(np.mean(macro_neg)) if macro_neg else 0,
        "macro_rand_mean": float(np.mean(macro_rand)) if macro_rand else 0
    }
    logger.info(f"[CORRELATION] pos={result['macro_pos_mean']:.4f}, neg={result['macro_neg_mean']:.4f}, rand={result['macro_rand_mean']:.4f}")
    return result

def monte_carlo_simulation(compute_fn, n_sims=1000, n_candles=500):
    logger.info(f"[MONTE CARLO] Starting {n_sims} simulations...")
    all_scores = []
    all_signals = []
    for sim in range(1, n_sims+1):
        if sim % 100 == 0:
            logger.info(f"[MONTE CARLO] {sim}/{n_sims} completed")
        corr = np.random.uniform(0.3, 0.95)
        corr_matrix = np.array([[1.0, corr, corr],
                                [corr, 1.0, corr],
                                [corr, corr, 1.0]])
        L = np.linalg.cholesky(corr_matrix)
        uncorr = np.random.normal(0, 1, (3, n_candles+100))
        correlated = L @ uncorr
        mu = np.random.uniform(-0.0002, 0.0002)
        sigma = np.random.uniform(0.005, 0.025)
        increments = mu + sigma * correlated
        price_sol = 100 * np.exp(np.cumsum(increments[0]))
        price_btc = 30000 * np.exp(np.cumsum(increments[1]))
        price_eth = 2000 * np.exp(np.cumsum(increments[2]))
        df_sol = pd.DataFrame({"close": price_sol, "high": price_sol*1.01, "low": price_sol*0.99})
        df_btc = pd.DataFrame({"close": price_btc, "high": price_btc*1.01, "low": price_btc*0.99})
        df_eth = pd.DataFrame({"close": price_eth, "high": price_eth*1.01, "low": price_eth*0.99})
        res = compute_fn(df_sol, df_btc, df_eth)
        all_scores.append(res.get(SCORE_KEY, 0.0))
        all_signals.append(res.get(SIGNAL_KEY, "NONE"))
    scores_arr = np.array(all_scores)
    long_cnt = sum(1 for s in all_signals if s == "LONG")
    short_cnt = sum(1 for s in all_signals if s == "SHORT")
    none_cnt = n_sims - long_cnt - short_cnt
    result = {
        "mean": float(np.mean(scores_arr)),
        "median": float(np.median(scores_arr)),
        "std": float(np.std(scores_arr)),
        "p5": float(np.percentile(scores_arr, 5)),
        "p95": float(np.percentile(scores_arr, 95)),
        "max": float(np.max(scores_arr)),
        "min": float(np.min(scores_arr)),
        "long_pct": long_cnt / n_sims,
        "short_pct": short_cnt / n_sims,
        "none_pct": none_cnt / n_sims,
        "scores": scores_arr.tolist()
    }
    logger.info(f"[MONTE CARLO] Completed: mean={result['mean']:.4f}, long={result['long_pct']*100:.1f}%")
    return result

def threshold_sweep(compute_fn, df_sol, df_btc, df_eth, thresholds, min_idx=100):
    logger.info("[THRESHOLD] Starting sweep...")
    scores = []
    min_len = min(len(df_sol), len(df_btc), len(df_eth))
    total = min_len - min_idx
    for i, idx in enumerate(range(min_idx, min_len), start=1):
        if i % 500 == 0:
            logger.info(f"[THRESHOLD] {i}/{total} processed")
        res = compute_fn(df_sol.iloc[:idx], df_btc.iloc[:idx], df_eth.iloc[:idx])
        scores.append(res.get(SCORE_KEY, 0.0))
    scores = np.array(scores)
    rows = []
    for th in thresholds:
        long = np.sum(scores > th)
        short = np.sum(scores < -th)
        none = len(scores) - long - short
        rows.append({
            "threshold": th,
            "long_pct": 100 * long / len(scores),
            "short_pct": 100 * short / len(scores),
            "none_pct": 100 * none / len(scores),
            "activation_pct": 100 * (long+short) / len(scores),
            "mean_score": np.mean(scores),
            "median_score": np.median(scores)
        })
    logger.info(f"[THRESHOLD] Completed")
    return pd.DataFrame(rows)

def scenario_testing(compute_fn):
    logger.info("[SCENARIO] Starting scenario tests...")
    scenarios = {
        "bull": {"trend": 0.0005, "vol": 0.01, "crash": False},
        "bear": {"trend": -0.0005, "vol": 0.01, "crash": False},
        "lateral": {"trend": 0.0, "vol": 0.002, "crash": False},
        "high_vol": {"trend": 0.0, "vol": 0.04, "crash": False},
        "low_vol": {"trend": 0.0, "vol": 0.001, "crash": False},
        "flash_crash": {"trend": 0.0, "vol": 0.01, "crash": True},
        "white_noise": {"trend": 0.0, "vol": 0.005, "crash": False}
    }
    results = []
    n = 1000
    for name, params in scenarios.items():
        logger.info(f"[SCENARIO] Running {name}...")
        rets = np.random.normal(params["trend"], params["vol"], n)
        if params.get("crash"):
            rets[300:350] = -0.05
        price = 100 * np.exp(np.cumsum(rets))
        df = pd.DataFrame({"close": price, "high": price*1.01, "low": price*0.99})
        scores = []
        signals = []
        for i in range(100, n):
            res = compute_fn(df.iloc[:i], df.iloc[:i], df.iloc[:i])
            scores.append(res.get(SCORE_KEY, 0.0))
            signals.append(res.get(SIGNAL_KEY, "NONE"))
        scores = np.array(scores)
        results.append({
            "scenario": name,
            "longs": sum(1 for s in signals if s == "LONG"),
            "shorts": sum(1 for s in signals if s == "SHORT"),
            "none": sum(1 for s in signals if s == "NONE"),
            "mean_score": np.mean(scores),
            "max_score": np.max(scores),
            "min_score": np.min(scores)
        })
    logger.info("[SCENARIO] Completed")
    return pd.DataFrame(results)

def stress_test(compute_fn):
    logger.info("[STRESS] Starting stress tests...")
    issues = []
    # 1. Constant prices
    const_price = np.ones(500) * 100.0
    df_const = pd.DataFrame({"close": const_price, "high": const_price+0.01, "low": const_price-0.01})
    try:
        for i in range(50, 500):
            compute_fn(df_const.iloc[:i], df_const.iloc[:i], df_const.iloc[:i])
    except Exception as e:
        issues.append(f"Constant price exception: {e}")
    # 2. ATR extremely small
    price_flat = 100 + np.random.normal(0, 0.0001, 500)
    df_flat = pd.DataFrame({"close": price_flat, "high": price_flat+0.0005, "low": price_flat-0.0005})
    try:
        for i in range(50, 500):
            compute_fn(df_flat.iloc[:i], df_flat.iloc[:i], df_flat.iloc[:i])
    except Exception as e:
        issues.append(f"ATR near zero exception: {e}")
    # 3. Gaps
    price_gap = 100 + np.cumsum(np.random.normal(0, 0.005, 500))
    price_gap[200] = price_gap[199] * 2.0
    df_gap = pd.DataFrame({"close": price_gap, "high": price_gap*1.02, "low": price_gap*0.98})
    try:
        for i in range(50, 500):
            compute_fn(df_gap.iloc[:i], df_gap.iloc[:i], df_gap.iloc[:i])
    except Exception as e:
        issues.append(f"Gap exception: {e}")
    # 4. Negative prices
    price_neg = np.ones(500) * 100.0
    price_neg[100] = -10.0
    df_neg = pd.DataFrame({"close": price_neg, "high": price_neg, "low": price_neg})
    try:
        for i in range(50, 500):
            compute_fn(df_neg.iloc[:i], df_neg.iloc[:i], df_neg.iloc[:i])
    except Exception as e:
        issues.append(f"Negative price exception: {e}")
    # 5. NaN values
    df_nan = df_neg.copy()
    df_nan.iloc[50, df_nan.columns.get_loc("close")] = np.nan
    try:
        compute_fn(df_nan.iloc[:200], df_nan.iloc[:200], df_nan.iloc[:200])
    except Exception as e:
        issues.append(f"NaN exception: {e}")
    # 6. Empty dataset
    df_empty = pd.DataFrame(columns=["close", "high", "low"])
    try:
        compute_fn(df_empty, df_empty, df_empty)
    except Exception as e:
        issues.append(f"Empty dataset exception: {e}")
    # 7. Too short dataset
    df_short = pd.DataFrame({"close": [100], "high": [101], "low": [99]})
    try:
        compute_fn(df_short, df_short, df_short)
    except Exception as e:
        issues.append(f"Too short dataset exception: {e}")
    logger.info(f"[STRESS] Completed with {len(issues)} issues")
    return issues

def generate_markdown_report(results: Dict) -> str:
    lines = []
    lines.append("# Signal Engine Research Report")
    lines.append(f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    lines.append("")
    lines.append("## Overview")
    lines.append(f"- Python version: {sys.version}")
    lines.append(f"- Random seed: {RANDOM_SEED}")
    lines.append(f"- Target candles: {TARGET_CANDLES}")
    lines.append(f"- Signal engine available: {SIGNAL_ENGINE_AVAILABLE}")
    lines.append("")
    if not SIGNAL_ENGINE_AVAILABLE:
        lines.append("## ERROR: signal_engine module not found")
        lines.append("The file `signal_engine.py` is missing or does not export `compute_signal`.")
        lines.append("All tests were skipped. Please ensure the module is present in the root directory.")
        lines.append("")
        return "\n".join(lines)
    lines.append("## Data Source")
    lines.append(f"- Real data used: {results.get('data_source', 'unknown')}")
    lines.append("")
    lines.append("## Integrity Test")
    if results["integrity"]["passed"]:
        lines.append("No integrity issues detected.")
    else:
        for issue in results["integrity"]["issues"]:
            lines.append(f"- {issue}")
    lines.append("")
    lines.append("## Mathematical Stability")
    stab = results["stability"]
    lines.append(f"- Valid scores generated: {len(stab['scores'])}")
    lines.append(f"- Macro values length: {len(stab['macro_vals'])}")
    if stab["errors"]:
        lines.append(f"- **Errors encountered:** {len(stab['errors'])}")
        for err in stab["errors"][:5]:
            lines.append(f"  - {err}")
    else:
        lines.append("- No NaN/infinite values detected.")
    lines.append("")
    lines.append("## ATR ≈ 0 Test")
    lines.append(f"- Result: {'PASS' if results['atr_zero']['passed'] else 'FAIL'}")
    if results['atr_zero']['errors']:
        for err in results['atr_zero']['errors']:
            lines.append(f"  - {err}")
    lines.append("")
    lines.append("## Extreme Correlation Test (Macro)")
    corr = results["extreme_corr"]
    lines.append(f"- Positive correlation macro mean: {corr['macro_pos_mean']:.4f}")
    lines.append(f"- Negative correlation macro mean: {corr['macro_neg_mean']:.4f}")
    lines.append(f"- Zero correlation macro mean: {corr['macro_rand_mean']:.4f}")
    lines.append("")
    lines.append("## Monte Carlo (1000 paths, correlated)")
    mc = results["monte_carlo"]
    lines.append(f"- Mean score: {mc['mean']:.4f}")
    lines.append(f"- Median score: {mc['median']:.4f}")
    lines.append(f"- Std dev: {mc['std']:.4f}")
    lines.append(f"- 5th percentile: {mc['p5']:.4f}, 95th: {mc['p95']:.4f}")
    lines.append(f"- LONG {mc['long_pct']*100:.1f}%, SHORT {mc['short_pct']*100:.1f}%, NONE {mc['none_pct']*100:.1f}%")
    lines.append("")
    lines.append("## Threshold Sweep (from real/synthetic data)")
    sweep_df = results["threshold_sweep"]
    lines.append("| threshold | long% | short% | none% | activation% | mean_score | median_score |")
    lines.append("|-----------|-------|--------|-------|-------------|------------|--------------|")
    for _, row in sweep_df.iterrows():
        lines.append(f"| {row['threshold']:.2f} | {row['long_pct']:.1f} | {row['short_pct']:.1f} | {row['none_pct']:.1f} | {row['activation_pct']:.1f} | {row['mean_score']:.4f} | {row['median_score']:.4f} |")
    lines.append("")
    lines.append("## Scenario Results")
    scenario_df = results["scenario"]
    lines.append("| scenario | longs | shorts | none | mean_score | max_score | min_score |")
    lines.append("|----------|-------|--------|------|------------|-----------|-----------|")
    for _, row in scenario_df.iterrows():
        lines.append(f"| {row['scenario']} | {row['longs']} | {row['shorts']} | {row['none']} | {row['mean_score']:.4f} | {row['max_score']:.4f} | {row['min_score']:.4f} |")
    lines.append("")
    lines.append("## Score Distribution (Real/Synthetic Data)")
    dist = results["score_distribution"]
    lines.append(f"- Mean: {dist['mean']:.4f}, Median: {dist['median']:.4f}")
    lines.append(f"- Std: {dist['std']:.4f}, 5th-95th: {dist['p5']:.4f} - {dist['p95']:.4f}")
    lines.append(f"- Max: {dist['max']:.4f}, Min: {dist['min']:.4f}")
    if results.get("histogram_path"):
        lines.append(f"- Histogram saved to `{results['histogram_path']}`")
    lines.append("")
    lines.append("## Stress Test Issues")
    if results["stress_issues"]:
        for issue in results["stress_issues"]:
            lines.append(f"- {issue}")
    else:
        lines.append("- No stress issues detected.")
    lines.append("")
    lines.append("## Performance Metrics")
    perf = results["performance"]
    lines.append(f"- Total execution time: {perf['total_time']:.2f} seconds")
    lines.append(f"- Integrity test: {perf['integrity_time']:.3f}s")
    lines.append(f"- Stability test: {perf['stability_time']:.3f}s")
    lines.append(f"- ATR zero test: {perf['atr_zero_time']:.3f}s")
    lines.append(f"- Extreme correlation: {perf['correlation_time']:.3f}s")
    lines.append(f"- Monte Carlo: {perf['monte_carlo_time']:.3f}s")
    lines.append(f"- Threshold sweep: {perf['threshold_time']:.3f}s")
    lines.append(f"- Scenario testing: {perf['scenario_time']:.3f}s")
    lines.append(f"- Stress test: {perf['stress_time']:.3f}s")
    lines.append("")
    lines.append("*End of Report*")
    return "\n".join(lines)

def generate_workflow_yaml():
    workflow_content = """name: Signal Research

on:
  workflow_dispatch:

jobs:
  research:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - run: |
          pip install numpy pandas requests matplotlib

      - run: |
          python run_all_research_tests.py

      - uses: actions/upload-artifact@v4
        with:
          name: reports
          path: reports/
"""
    workflow_path = ".github/workflows/run_research.yml"
    os.makedirs(os.path.dirname(workflow_path), exist_ok=True)
    with open(workflow_path, "w") as f:
        f.write(workflow_content)
    logger.info(f"Workflow YAML written to {workflow_path}")

# ==================== MAIN ====================
def main():
    logger.info("=" * 70)
    logger.info(" EMPIRICAL AUDIT OF signal_engine.py")
    logger.info("=" * 70)
    start_total = time.time()

    # Generate workflow YAML
    generate_workflow_yaml()

    # Introspect signal engine (if available)
    compute_fn, engine_keys = introspect_signal_engine()
    if compute_fn is None:
        logger.error("Cannot proceed without signal_engine. Creating error report.")
        # Create empty results directory and error report
        error_report = "# Signal Engine Research Report\n\n"
        error_report += "## ERROR: signal_engine module not found\n"
        error_report += "The file `signal_engine.py` is missing or does not export `compute_signal`.\n"
        error_report += "Please add the module and rerun the research.\n"
        report_path = os.path.join(OUTPUT_DIR, "signal_research_report.md")
        with open(report_path, "w") as f:
            f.write(error_report)
        # Also create empty CSV files to avoid missing artifact issues
        for name in ["threshold_sweep.csv", "scenario_results.csv", "score_distribution.csv", "stress_results.csv", "performance_metrics.csv"]:
            pd.DataFrame().to_csv(os.path.join(OUTPUT_DIR, name), index=False)
        logger.info("Error report and empty artifacts generated.")
        return

    # Load data
    df_sol, df_btc, df_eth = load_real_data()
    data_source = "Binance real data" if not any(df is None for df in [df_sol, df_btc, df_eth]) else "Synthetic"
    logger.info(f"Data source: {data_source}")
    logger.info(f"Shapes: SOL={df_sol.shape}, BTC={df_btc.shape}, ETH={df_eth.shape}")

    results = {"data_source": data_source}

    # Run tests
    try:
        t0 = time.time()
        results["integrity"] = test_integrity(df_sol, df_btc, df_eth)
        integrity_time = time.time() - t0
    except Exception as e:
        logger.exception("Integrity test failed")
        results["integrity"] = {"passed": False, "issues": [str(e)]}
        integrity_time = time.time() - t0

    try:
        t0 = time.time()
        results["stability"] = test_mathematical_stability(compute_fn, df_sol, df_btc, df_eth, engine_keys)
        stability_time = time.time() - t0
    except Exception as e:
        logger.exception("Stability test failed")
        results["stability"] = {"scores": [], "macro_vals": [], "signals": [], "errors": [str(e)]}
        stability_time = time.time() - t0

    try:
        t0 = time.time()
        results["atr_zero"] = test_atr_zero(compute_fn)
        atr_zero_time = time.time() - t0
    except Exception as e:
        logger.exception("ATR zero test failed")
        results["atr_zero"] = {"passed": False, "errors": [str(e)]}
        atr_zero_time = time.time() - t0

    try:
        t0 = time.time()
        results["extreme_corr"] = test_extreme_correlation(compute_fn)
        correlation_time = time.time() - t0
    except Exception as e:
        logger.exception("Extreme correlation test failed")
        results["extreme_corr"] = {"macro_pos_mean": 0, "macro_neg_mean": 0, "macro_rand_mean": 0}
        correlation_time = time.time() - t0

    try:
        t0 = time.time()
        results["monte_carlo"] = monte_carlo_simulation(compute_fn, n_sims=1000, n_candles=500)
        monte_carlo_time = time.time() - t0
    except Exception as e:
        logger.exception("Monte Carlo failed")
        results["monte_carlo"] = {"mean": 0, "median": 0, "std": 0, "p5": 0, "p95": 0, "max": 0, "min": 0, "long_pct": 0, "short_pct": 0, "none_pct": 1, "scores": []}
        monte_carlo_time = time.time() - t0

    try:
        t0 = time.time()
        thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
        sweep_df = threshold_sweep(compute_fn, df_sol, df_btc, df_eth, thresholds, min_idx=100)
        sweep_df.to_csv(os.path.join(OUTPUT_DIR, "threshold_sweep.csv"), index=False)
        results["threshold_sweep"] = sweep_df
        threshold_time = time.time() - t0
    except Exception as e:
        logger.exception("Threshold sweep failed")
        results["threshold_sweep"] = pd.DataFrame()
        threshold_time = time.time() - t0

    try:
        t0 = time.time()
        scenario_df = scenario_testing(compute_fn)
        scenario_df.to_csv(os.path.join(OUTPUT_DIR, "scenario_results.csv"), index=False)
        results["scenario"] = scenario_df
        scenario_time = time.time() - t0
    except Exception as e:
        logger.exception("Scenario testing failed")
        results["scenario"] = pd.DataFrame()
        scenario_time = time.time() - t0

    try:
        scores_all = results["stability"].get("scores", [])
        if scores_all:
            scores_arr = np.array(scores_all)
            results["score_distribution"] = {
                "mean": float(np.mean(scores_arr)),
                "median": float(np.median(scores_arr)),
                "std": float(np.std(scores_arr)),
                "p5": float(np.percentile(scores_arr, 5)),
                "p95": float(np.percentile(scores_arr, 95)),
                "max": float(np.max(scores_arr)),
                "min": float(np.min(scores_arr))
            }
            pd.DataFrame([results["score_distribution"]]).to_csv(os.path.join(OUTPUT_DIR, "score_distribution.csv"), index=False)
            if HAS_PLT:
                hist_path = os.path.join(OUTPUT_DIR, "score_distribution.png")
                plt.figure(figsize=(10,6))
                plt.hist(scores_arr, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
                plt.title("Signal Score Distribution")
                plt.xlabel("Score")
                plt.ylabel("Frequency")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(hist_path)
                plt.close()
                results["histogram_path"] = hist_path
        else:
            results["score_distribution"] = {"mean": 0, "median": 0, "std": 0, "p5": 0, "p95": 0, "max": 0, "min": 0}
    except Exception as e:
        logger.exception("Score distribution failed")
        results["score_distribution"] = {"mean": 0, "median": 0, "std": 0, "p5": 0, "p95": 0, "max": 0, "min": 0}

    try:
        t0 = time.time()
        results["stress_issues"] = stress_test(compute_fn)
        stress_time = time.time() - t0
        with open(os.path.join(OUTPUT_DIR, "stress_results.csv"), "w") as f:
            f.write("issue\n")
            for issue in results["stress_issues"]:
                f.write(f'"{issue}"\n')
    except Exception as e:
        logger.exception("Stress test failed")
        results["stress_issues"] = [str(e)]
        stress_time = time.time() - t0

    # Performance metrics
    total_time = time.time() - start_total
    results["performance"] = {
        "total_time": total_time,
        "integrity_time": integrity_time,
        "stability_time": stability_time,
        "atr_zero_time": atr_zero_time,
        "correlation_time": correlation_time,
        "monte_carlo_time": monte_carlo_time,
        "threshold_time": threshold_time,
        "scenario_time": scenario_time,
        "stress_time": stress_time
    }
    perf_df = pd.DataFrame([results["performance"]])
    perf_df.to_csv(os.path.join(OUTPUT_DIR, "performance_metrics.csv"), index=False)

    # Generate report
    report = generate_markdown_report(results)
    report_path = os.path.join(OUTPUT_DIR, "signal_research_report.md")
    with open(report_path, "w") as f:
        f.write(report)

    logger.info("=" * 70)
    logger.info(f"AUDIT COMPLETED. Reports saved in '{OUTPUT_DIR}'")
    logger.info("=" * 70)

if __name__ == "__main__":
    main()
