import yfinance as yf
from models.valuation_result import ValuationResult, Valuation, StockInfo, NetProfitHistory, AverageGrowth
import requests
import pandas as pd
from tradingview_screener import Query, col
import time

def get_cik_from_ticker(ticker: str) -> str:
    ticker = ticker.upper()
    mapping_url = "https://www.sec.gov/files/company_tickers.json"
    headers = {"User-Agent": "Your Name Contact@example.com"}  # Required by SEC
    mapping = requests.get(mapping_url, headers=headers).json()

    for item in mapping.values():  # iterate over dict values
        if item['ticker'].upper() == ticker:
            return str(item['cik_str']).zfill(10)
    print(f"Ticker {ticker} not found in SEC mapping.")
    return None


def get_net_income(ticker: str) -> list[NetProfitHistory]:
    cik = get_cik_from_ticker(ticker)
    if  cik is not None: 
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
        headers = {"User-Agent": "Your Name Contact@example.com"}
        try:
            data = requests.get(url, headers=headers).json()
            if "facts" in data and "us-gaap" in data["facts"]:
                if "NetIncomeLoss" in data["facts"]["us-gaap"]:
                    # Extract Net Income (US-GAAP: NetIncomeLoss)
                    facts = data["facts"]["us-gaap"]["NetIncomeLoss"]["units"]["USD"]

                    # Convert to DataFrame
                    df = pd.DataFrame(facts)
                    df = df[["fy", "fp", "end", "val"]]
                    df = df[df["fp"] == "FY"]   # only full year reports

                    # Keep the latest report for each year
                    df = df.sort_values(["fy", "end"], ascending=[False, False])
                    df = df.groupby("fy").first().reset_index()

                    # Keep last 6 years
                    df = df.sort_values("fy", ascending=False).head(6)
                    df = df.sort_values("fy", ascending=True)  # ascending for consistency

                    # Convert to list of NetProfitHistory
                    return [
                        NetProfitHistory(year=int(row.fy), value=float(row.val))
                        for row in df.itertuples(index=False)
                    ]
                else:
                    return None
            else:
                return None
        except requests.exceptions.JSONDecodeError:
            return None
    else:
        return None
    
def get_net_income_from_file(ticker: str, exchange: str) -> list[NetProfitHistory]:
    net_profits = pd.read_csv('net_income_nyse_nasdaq.csv')
    net_profits_for_ticker = net_profits[(net_profits['tickers'] == ticker) & (net_profits['exchange'] == exchange)]

    if net_profits_for_ticker.empty:
        return None

    row = net_profits_for_ticker.iloc[0]  # get the first (and probably only) row
    result: list[NetProfitHistory] = []

    # Iterate over columns that are years
    for column in net_profits_for_ticker.columns:
        if column.isdigit():  # only process year columns
            value = row[column]
            if pd.notna(value):  # skip NaN values
                result.append(NetProfitHistory(year=int(column), value=float(value)))

    return result

def calculate_average_growth(history: list[NetProfitHistory]) -> AverageGrowth:
    # ensure chronological order
    history = sorted(history, key=lambda x: x.year)

    if len(history) < 2:
        return AverageGrowth(ttm=None, threeYears=None, fiveYears=None)

    # compute YoY growths
    yoy_growths = []
    for i in range(1, len(history)):
        prev, curr = history[i-1].value, history[i].value
        if prev and prev != 0:
            yoy_growths.append((curr - prev) / prev)

    # last year's YoY growth (TTM)
    ttm_growth = yoy_growths[-1] if len(yoy_growths) >= 1 else None

    # average of last 3 YoY growths
    three_years_growth = (sum(yoy_growths[-3:]) / 3) if len(yoy_growths) >= 3 else None

    # average of last 5 YoY growths
    five_years_growth = (sum(yoy_growths[-5:]) / 5) if len(yoy_growths) >= 5 else None

    return AverageGrowth(
        ttm=round(ttm_growth*100,2) if ttm_growth is not None else None,
        threeYears=round(three_years_growth*100,2) if three_years_growth is not None else None,
        fiveYears=round(five_years_growth*100,2) if five_years_growth is not None else None
    )

def safe_float(val):
    return float(val) if val is not None else None

def get_valuation(exchange:str, ticker: str) -> ValuationResult:
    stockInfo = None
    max_retries = 2
    # --- Try Yahoo Finance ---
    retries = 0
    while retries < max_retries and stockInfo is None:
        try:
            stock_yf = yf.Ticker(ticker)
            info = stock_yf.info

            # Sometimes .info is empty even if no exception is raised
            if info and "shortName" in info:
                stockInfo = StockInfo(
                    name=info.get("shortName"),
                    ticker=ticker,
                    exchange=exchange,
                    price=safe_float(info.get("previousClose")),
                    country=info.get("country"),
                    capitalization=safe_float(info.get("marketCap")),
                    sector=info.get("sector"),
                    industry=info.get("industry"),
                    epsTtm=safe_float(info.get("epsTrailingTwelveMonths")),
                    peTtm=safe_float(info.get("trailingPE")),
                    dividendYield=safe_float(info.get("dividendYield")),
                )
                break
        except Exception as e:
            retries += 1
            time.sleep(1.5 * retries)  # exponential backoff

    # --- Fallback to TradingView ---
    if stockInfo is None:
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
            )
            .where(col("name") == ticker)
            .get_scanner_data()
        )

        row = df[1].iloc[0]

        stockInfo = StockInfo(
            name=row["description"],
            ticker=row["ticker"],
            exchange=row["exchange"],
            price=safe_float(row["close"]),
            country=row["country"],
            capitalization=safe_float(row["market_cap_basic"]),
            sector=row["sector"],
            industry=row["industry"],
            epsTtm=safe_float(row["earnings_per_share_basic_ttm"]),
            peTtm=safe_float(row["price_earnings_ttm"]),
            dividendYield=safe_float(row["dividends_yield"]) * 100
            if row["dividends_yield"] is not None
            else None,
        )

    netProfitHistory = get_net_income_from_file(ticker, exchange)
    print(ticker)
    if netProfitHistory is not None and len(netProfitHistory)>=5:
        averageGrowth: AverageGrowth = calculate_average_growth(netProfitHistory)
    else:
        averageGrowth: AverageGrowth = None
    if averageGrowth is not None and averageGrowth.fiveYears is not None and stockInfo.epsTtm is not None:
        fairPrice = averageGrowth.fiveYears* stockInfo.epsTtm
    else:
        fairPrice = None
    if averageGrowth is not None:
        explanationText = f"{stockInfo.epsTtm} x {averageGrowth.fiveYears} = {fairPrice}"
    else:
        explanationText = ""
    if averageGrowth is not None and averageGrowth.fiveYears is not None and stockInfo.epsTtm is not None:
        resultPercent = round((fairPrice/stockInfo.price)-1,2)*100
    else:
        resultPercent = 0.0
    resultLabel = "Overvalued"
    if(resultPercent>0):
        resultLabel = "Undervalued"
    else:
        resultLabel = "Overvalued"
    valuation = Valuation(
        fairPrice=fairPrice,
        resultPercent=resultPercent,
        resultLabel=resultLabel,
        formula=explanationText,
        explanation="",
        netProfitHistory=netProfitHistory,
        avgGrowth=averageGrowth)
    result  = ValuationResult(stockInfo=stockInfo, valuation=valuation)
    return result
