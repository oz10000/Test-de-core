# signal_engine.py
# PyDROID SOLANA CORE Ω – Deterministic Signal Engine
# Validated, frozen, ready for production.

import numpy as np
import pandas as pd
import math
from typing import Dict

# ==================== PARÁMETROS CONGELADOS ====================
EMA_PERIOD = 20
ADX_PERIOD = 14
ATR_PERIOD = 14
CORR_WINDOW = 50
ADX_THRESHOLD = 25
SIGMOID_SCALE = 10.0
SCORE_THRESHOLD = 0.15
MIN_CANDLES = max(EMA_PERIOD, ADX_PERIOD, ATR_PERIOD, CORR_WINDOW) + 10

# ==================== INDICADORES ROBUSTOS ====================
def ema(series: pd.Series, period: int) -> pd.Series:
    """EMA con manejo de NaN"""
    return series.ewm(span=period, adjust=False).mean()

def atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    """ATR seguro (true range)"""
    high = df['high']
    low = df['low']
    close = df['close']
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def adx_wilder(df: pd.DataFrame, period: int = ADX_PERIOD) -> pd.Series:
    """
    ADX según Wilder (método clásico).
    Retorna Serie con ADX, sin valores NaN al inicio.
    """
    high = df['high']
    low = df['low']
    close = df['close']
    
    # True Range
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    
    # +DM, -DM
    up_move = high.diff()
    down_move = low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Suavizado exponencial estilo Wilder (α = 1/period)
    atr_series = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean() / atr_series
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean() / atr_series
    
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    return adx

def compute_micro_signal(df_sol: pd.DataFrame) -> float:
    """Micro = pendiente EMA20 / ATR. Retorna 0 si datos insuficientes."""
    if len(df_sol) < EMA_PERIOD + 2:
        return 0.0
    ema_series = ema(df_sol['close'], EMA_PERIOD)
    slope = ema_series.iloc[-1] - ema_series.iloc[-2]
    atr_val = atr(df_sol).iloc[-1]
    if np.isnan(atr_val) or atr_val == 0:
        return 0.0
    return slope / atr_val

def compute_regime(df_sol: pd.DataFrame) -> float:
    """Regime = 1 si ADX > umbral, sino 0."""
    if len(df_sol) < ADX_PERIOD + 1:
        return 0.0
    adx_series = adx_wilder(df_sol, ADX_PERIOD)
    adx_val = adx_series.iloc[-1]
    if np.isnan(adx_val):
        return 0.0
    return 1.0 if adx_val > ADX_THRESHOLD else 0.0

def compute_macro_signal(df_btc: pd.DataFrame, df_eth: pd.DataFrame, df_sol: pd.DataFrame) -> float:
    """
    Macro signal = alignment * sigmoid(10 * (mean_corr - 0.5))
    con clipping y protección contra NaN.
    """
    if len(df_btc) < EMA_PERIOD + 1 or len(df_eth) < EMA_PERIOD + 1 or len(df_sol) < CORR_WINDOW:
        return 0.0
    
    # Pendientes de EMA
    ema_btc = ema(df_btc['close'], EMA_PERIOD)
    ema_eth = ema(df_eth['close'], EMA_PERIOD)
    btc_slope = ema_btc.iloc[-1] - ema_btc.iloc[-2]
    eth_slope = ema_eth.iloc[-1] - ema_eth.iloc[-2]
    
    alignment = 1.0 if (btc_slope * eth_slope > 0) else 0.0
    if alignment == 0.0:
        return 0.0
    
    # Correlaciones rolling (última ventana)
    btc_close = df_btc['close'].iloc[-CORR_WINDOW:]
    eth_close = df_eth['close'].iloc[-CORR_WINDOW:]
    sol_close = df_sol['close'].iloc[-CORR_WINDOW:]
    corr_btc = btc_close.corr(sol_close)
    corr_eth = eth_close.corr(sol_close)
    
    if np.isnan(corr_btc) or np.isnan(corr_eth):
        return 0.0
    
    mean_corr = (corr_btc + corr_eth) / 2.0
    # Sigmoid seguro
    z = SIGMOID_SCALE * (mean_corr - 0.5)
    if z > 50:
        macro = 1.0
    elif z < -50:
        macro = 0.0
    else:
        macro = 1.0 / (1.0 + math.exp(-z))
    return float(np.clip(macro, 0.0, 1.0))

def compute_signal(df_sol: pd.DataFrame, df_btc: pd.DataFrame, df_eth: pd.DataFrame) -> Dict:
    """
    Retorna diccionario con:
        signal: 'LONG', 'SHORT', 'NONE'
        score: float entre -1 y 1
        micro: float
        macro: float
        regime: int
        adx: float (valor ADX real)
    """
    # Validación de tamaño mínimo
    if len(df_sol) < MIN_CANDLES or len(df_btc) < MIN_CANDLES or len(df_eth) < MIN_CANDLES:
        return {
            "signal": "NONE",
            "score": 0.0,
            "micro": 0.0,
            "macro": 0.0,
            "regime": 0,
            "adx": 0.0
        }
    
    micro = compute_micro_signal(df_sol)
    regime = compute_regime(df_sol)
    macro = compute_macro_signal(df_btc, df_eth, df_sol)
    
    # ADX real para diagnóstico
    adx_val = adx_wilder(df_sol, ADX_PERIOD).iloc[-1]
    
    raw_score = micro * regime * macro
    score = math.tanh(raw_score)   # acotado a [-1,1]
    
    # Decisión
    if score > SCORE_THRESHOLD:
        signal = "LONG"
    elif score < -SCORE_THRESHOLD:
        signal = "SHORT"
    else:
        signal = "NONE"
    
    return {
        "signal": signal,
        "score": score,
        "micro": micro,
        "macro": macro,
        "regime": regime,
        "adx": adx_val
    }
