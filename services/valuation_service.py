import yfinance as yf
from models.valuation_result import ValuationResult, Valuation, StockInfo, NetProfitHistory, AverageGrowth
import requests
import pandas as pd
from tradingview_screener import Query, col
import time
import apimoex

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
    #TODO: differentiate between MOEX and the rest, 'moex_data.csv'
    if exchange == 'MOEX':
        net_profits = pd.read_csv('moex_data.csv')
        net_profits_for_ticker = net_profits[net_profits['Ticker'] == ticker]
        net_profits_for_ticker = net_profits_for_ticker[['2019','2020','2021','2022','2023','2024']]
    else:
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
    result.sort(key=lambda r: r.year)
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
    #ttm_cagr = calculate_cagr(history[-2].value, history[-1].value, 1) if len(yoy_growths) >= 1 else None
    #ttm_growth = ttm_cagr

    # average of last 3 YoY growths
    three_years_growth = (sum(yoy_growths[-3:]) / 3) if len(yoy_growths) >= 3 else None
    #three_years_cagr = calculate_cagr(history[-3].value ,history[-1].value, 3) if len(yoy_growths) >= 3 else None
    #three_years_growth = three_years_cagr

    # average of last 5 YoY growths
    five_years_growth = (sum(yoy_growths[-5:]) / 5) if len(yoy_growths) >= 4 else None
    #five_years_cagr = calculate_cagr(history[-5].value, history[-1].value, 5) if len(yoy_growths) >= 4 else None
    #five_years_growth = five_years_cagr
    return AverageGrowth(
        ttm=round(ttm_growth*100,2) if ttm_growth is not None else None,
        threeYears=round(three_years_growth*100,2) if three_years_growth is not None else None,
        fiveYears=round(five_years_growth*100,2) if five_years_growth is not None else None
    )

def safe_float(val):
    return float(val) if val is not None else None

def get_price_from_moex(ticker:str) -> float:
    with requests.Session() as session:
        data = apimoex.get_board_history(session, ticker)
        df = pd.DataFrame(data)
        df.set_index('TRADEDATE', inplace=True)
        return float(df[df['BOARDID']=='TQBR'].iloc[-1]['CLOSE'])

def get_moex_stock_data(ticker:str) -> tuple:
    data = pd.read_csv('moex_data.csv')
    df = pd.DataFrame(columns=['ticker', 'name', 'description', 'exchange', 'close', 'country',
       'market_cap_basic', 'sector', 'industry',
       'earnings_per_share_basic_ttm', 'price_earnings_ttm', 'dividends_yield',
       'free_cash_flow_fy', 'debt_to_equity', 'sector_ru', 'industry_ru'])
    data = data[data["Ticker"]==ticker] 
    name = data.iloc[0]['Name']
    description = ''
    exchange = 'MOEX'
    close = get_price_from_moex(ticker=ticker)
    country = 'Russia'
    market_cap_basic = round(close * float(data.iloc[0]['Issue']),2)
    sector = data.iloc[0]['Sector']
    industry = data.iloc[0]['Industry']
    earnings_per_share_basic_ttm = data.iloc[0]['EPS']
    price_earnings_ttm = round(close/earnings_per_share_basic_ttm,2)
    dividends_yield = data.iloc[0]['Dividends']
    free_cash_flow_fy = data.iloc[0]['FCF']
    #because equity here isn't equity these are assets I am cacling DE as D/(A-D):
    debt_to_equity = round(abs(data.iloc[0]['Debt']) / (data.iloc[0]['Equity']-abs(data.iloc[0]['Debt'])),2)
    sector_ru = data.iloc[0]['SectorRu']
    industry_ru = data.iloc[0]['IndustryRu']
    df.loc[len(df)] = [ticker, name, description, exchange, close, country, market_cap_basic, sector, industry, 
                       earnings_per_share_basic_ttm, price_earnings_ttm, dividends_yield, free_cash_flow_fy,
                       debt_to_equity, sector_ru, industry_ru]
    return 1,df

def get_valuation(exchange:str, ticker: str) -> ValuationResult:
    #TODO: if exchange is moex logic is different
    stockInfo = None
    if stockInfo is None:
        df = pd.DataFrame()
        if exchange == 'MOEX':
            df = get_moex_stock_data(ticker = ticker)
        else:
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
                .where(col("name") == ticker)
                .get_scanner_data()
            )
        if len(df[1])>0:
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
                sectorRu= row["sector_ru"]
                if exchange=="MOEX"
                else None,
                industryRu= row["industry_ru"]
                if exchange=="MOEX"
                else None,
                epsTtm=safe_float(row["earnings_per_share_basic_ttm"]),
                peTtm=safe_float(row["price_earnings_ttm"]),
                dividendYield=safe_float(row["dividends_yield"])
                if row["dividends_yield"] is not None
                else None,
                freeCashFlow=safe_float(row["free_cash_flow_fy"]) 
                if row["dividends_yield"] is not None
                else None,
                debtToEquity=safe_float(row["debt_to_equity"]) 
                if row["dividends_yield"] is not None
                else None
            )

            netProfitHistory = get_net_income_from_file(ticker, exchange)
            print(ticker)
            if netProfitHistory is not None and len(netProfitHistory)>=5:
                averageGrowth: AverageGrowth = calculate_average_growth(netProfitHistory)
            else:
                averageGrowth: AverageGrowth = None
            peg=1
            #if averageGrowth is not None and averageGrowth.fiveYears is not None and stockInfo.peTtm is not None:
                #peg = round(stockInfo.peTtm/averageGrowth.fiveYears,2)
                #if peg < 0.01:
                    #peg = 1
            if averageGrowth is not None and averageGrowth.fiveYears is not None and averageGrowth.fiveYears > 25:
                calcAvgGrowthRate = 25.00
            elif averageGrowth is not None and averageGrowth.fiveYears is not None and averageGrowth.fiveYears <= 25:
                calcAvgGrowthRate = averageGrowth.fiveYears
            else:
                calcAvgGrowthRate = None
            if averageGrowth is not None and averageGrowth.fiveYears is not None and stockInfo.epsTtm is not None:
                fairPrice = calcAvgGrowthRate* stockInfo.epsTtm*peg
            else:
                fairPrice = None
            if averageGrowth is not None and stockInfo.epsTtm is not None and fairPrice is not None:
                explanationText = f"{round(calcAvgGrowthRate,2)} x {round(stockInfo.epsTtm,2)} x 1 = {round(fairPrice,2)}"
            else:
                explanationText = ""
            if averageGrowth is not None and averageGrowth.fiveYears is not None and stockInfo.epsTtm is not None:
                resultPercent = round(((round(fairPrice,2)-round(stockInfo.price,2))/round(stockInfo.price,2))*100,2)
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
                avgGrowth=averageGrowth,
                peg=peg
                )
            result  = ValuationResult(stockInfo=stockInfo, valuation=valuation)
            return result
        else:
            return None

def calculate_cagr(beginning_value, ending_value, number_of_years):
    """
    Calculates the Compound Annual Growth Rate (CAGR).

    Args:
    beginning_value (float): The initial value of the investment or metric.
    ending_value (float): The final value of the investment or metric.
    number_of_years (int or float): The number of periods (years) over which 
                                        the growth occurred.

    Returns:
    float: The Compound Annual Growth Rate (CAGR).
    """
    if number_of_years <= 0:
        raise ValueError("Number of years must be greater than zero.")

    cagr = (ending_value / beginning_value) ** (1 / number_of_years) - 1
    return cagr