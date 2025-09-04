import yfinance as yf
from models.valuation_result import ValuationResult, Valuation, StockInfo, NetProfitHistory, AverageGrowth
import requests
import pandas as pd

def get_cik_from_ticker(ticker: str) -> str:
    ticker = ticker.upper()
    mapping_url = "https://www.sec.gov/files/company_tickers.json"
    headers = {"User-Agent": "Your Name Contact@example.com"}  # Required by SEC
    mapping = requests.get(mapping_url, headers=headers).json()

    for item in mapping.values():  # iterate over dict values
        if item['ticker'].upper() == ticker:
            return str(item['cik_str']).zfill(10)
    raise ValueError(f"Ticker {ticker} not found in SEC mapping.")


def get_net_income(ticker: str) -> list[NetProfitHistory]:
    cik = get_cik_from_ticker(ticker)  
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    headers = {"User-Agent": "Your Name Contact@example.com"}
    data = requests.get(url, headers=headers).json()

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
        ttm=round(ttm_growth*100,2),
        threeYears=round(three_years_growth*100,2),
        fiveYears=round(five_years_growth*100,2)
    )

def get_valuation(exchange:str, ticker: str) -> ValuationResult:
    stock = yf.Ticker(ticker=ticker)
    stockInfo: StockInfo = StockInfo(
        name=stock.info.get("shortName"),
        ticker=ticker,
        exchange=exchange,
        price=stock.info.get("previousClose"),
        country=stock.info.get("country"),
        capitalization=float(stock.info.get("marketCap")),
        sector=stock.info.get("sector"),
        industry=stock.info.get("industry"),
        epsTtm=stock.info.get("epsTrailingTwelveMonths"),
        peTtm=stock.info.get("trailingPE"),
        dividendYield=stock.info.get("dividendYield")
        )
    netProfitHistory = get_net_income(ticker)
    averageGrowth: AverageGrowth = calculate_average_growth(netProfitHistory)
    fairPrice = averageGrowth.fiveYears* stockInfo.epsTtm
    explanationText = f"{stockInfo.epsTtm} x {averageGrowth.fiveYears} = {fairPrice}"
    resultPercent = round((fairPrice/stockInfo.price)-1,2)*100
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
