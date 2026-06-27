import pandas as pd
from tradingview_screener import Query, col
import time
import apimoex
import requests

def cache_nyse_data():
    df = (
            Query()
            .select(
                "name",
                "description",
                "exchange",
                "close",
                "country",
                "market_cap_basic",
                "sector",
                "industry",
                "earnings_per_share_basic_ttm",
                "price_earnings_ttm",
                "dividends_yield",
                "free_cash_flow_fy",
                "debt_to_equity"
            )
            .where(col("exchange") == 'NYSE')
            .limit(1_000_000)
            .get_scanner_data()
        )
    df[1].to_csv('nyse_cache.csv')
    
def cache_nasdaq_data():
    df = (
            Query()
            .select(
                "name",
                "description",
                "exchange",
                "close",
                "country",
                "market_cap_basic",
                "sector",
                "industry",
                "earnings_per_share_basic_ttm",
                "price_earnings_ttm",
                "dividends_yield",
                "free_cash_flow_fy",
                "debt_to_equity"
            )
            .where(col("exchange") == 'NASDAQ')
            .limit(1_000_000)
            .get_scanner_data()
        )
    df[1].to_csv('nasdaq_cache.csv')
    
def cache_moex_data():
    file = pd.read_csv('moex_data.csv')
    moex_tickers = file['Ticker']   
    tickers = moex_tickers
    results = {}

    with requests.Session() as session:
        for ticker in tickers:
            # Fetch only the single last day of history to keep it lightweight
            candles = apimoex.get_board_history(session, ticker, start="2026-06-01")
            
            if candles:
                # Grab the 'close' price from the very last entry in the list
                last_close = candles[-1]["CLOSE"]
                results[ticker] = last_close
                print(f"Loaded {ticker}: {last_close}")
            else:
                print(f"Could not find data for {ticker}")
                
            # Introduce a 1.5-second delay between requests to remain stealthy
            time.sleep(1.5)

    # Convert results dict to a clean DataFrame
    df_closes = pd.DataFrame(list(results.items()), columns=["Ticker", "Last_Close"])
    df_closes.to_csv('moex_cache.csv')
    