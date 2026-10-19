"""
Trend-following and momentum indicators

"""

import numpy as np
import pandas as pd


def sma(price, window=20):
    """Simple moving average
    price: Pandas Series
    window: str default 20 days    
    """
    return price.rolling(window=window, min_periods=window).mean()


def ema(price, window=20):
    """Exponential moving average, initialised with the first price.
    price: Pandas Series
    window: str default 20 days      
    """
    return price.ewm(
        span=window,
        adjust=False,
        min_periods=window
        ).mean()


def roc(price, window=12):
    """Percentage price change over the specified window.
    price: Pandas Series
    window: str default 12 days  
    """
    return 100 * (price / price.shift(window) - 1)


def rsi(price, window=14):
    """Wilder's RSI, initialised with mean gains and losses.
    Requires prices without missing values.
    Returns 50 for a completely flat window.
    price: Pandas Series
    window: str default 14 days  
    """
    if window < 1:
        raise ValueError("window must be at least 1.")

    price = price.astype(float)

    if price.isna().any():
        raise ValueError("Remove or handle missing prices before calculating RSI.")

    result = pd.Series(np.nan, index=price.index, name="RSI")

    if len(price) <= window:
        return result

    change = price.diff()
    gains = change.clip(lower=0)
    losses = -change.clip(upper=0)

    #Initial averages use the first 'window' price changes.
    avg_gain = gains.iloc[1:window + 1].mean()
    avg_loss = losses.iloc[1:window + 1].mean()

    def score(gain, loss):
        if gain == 0 and loss == 0:
            return 50.0
        return 100 * gain / (gain + loss)

    result.iloc[window] = score(avg_gain, avg_loss)

    #Wilder's recursive smoothing: alpha = 1 / window.
    for i in range(window + 1, len(price)):
        avg_gain = ((window - 1) * avg_gain + gains.iloc[i]) / window
        avg_loss = ((window - 1) * avg_loss + losses.iloc[i]) / window
        result.iloc[i] = score(avg_gain, avg_loss)

    return result


def macd(price, fast=12, slow=26, signal=9):
    """Return MACD, signal line and histogram as a DataFrame.
    price: Pandas Series
    fast: str default 12 days  
    slow: str default 26 days  
    signal: str default 9 days      
    """
    if not 1 <= fast < slow or signal < 1:
        raise ValueError("Require 1 <= fast < slow and signal >= 1.")

    macd_line = ema(price, fast) - ema(price, slow)
    signal_line = ema(macd_line, signal)

    return pd.DataFrame({
        "MACD": macd_line,
        "MACD_signal": signal_line,
        "MACD_histogram": macd_line - signal_line
    })