"""
Yahoo Finance download

"""

import yfinance as yf


def download(tickers: list, period: str = "5y", interval: str = "1d"):
    """
    Download daily prices for the last five years
    tickers: list of tickers
    period: str with default 5 years
    interval: str with default daily
    Downloaded data are saved as csv file
    """
    data = yf.download(
        tickers = tickers,
        period = period,
        interval = interval,
        auto_adjust = False
    )
        # Extract closing prices: one column per stock
    closing_prices = data["Close"].reindex(columns=tickers)
    closing_prices.index.name = "Date"
    
    # Inspect and save
    print(closing_prices.head())
    print(closing_prices.tail())
    
    closing_prices.to_csv("Data.csv")