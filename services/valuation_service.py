import yfinance as yf
from models.valuation_result import ValuationResult, Valuation, StockInfo, NetProfitHistory, AverageGrowth
import requests
import numpy as np
import pandas as pd
from tradingview_screener import Query, col
import yfinance as yf
import statsmodels.api as sm
import time
import apimoex
from datetime import datetime, timedelta

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
    if exchange == 'MOEX':
        net_profits = pd.read_csv('moex_data.csv')
        net_profits_for_ticker = net_profits[net_profits['Ticker'] == ticker]
        last_five_years = [str(datetime.now().year - i) for i in range(1,7)]
        net_profits_for_ticker = net_profits_for_ticker[last_five_years]
        net_profits_for_ticker = net_profits_for_ticker.dropna(axis=1)
        net_profits_for_ticker = net_profits_for_ticker.iloc[:, :5]
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


def get_prices_from_moex(ticker:str, boardid:str, market: str) -> pd.DataFrame:
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=3*365)).strftime('%Y-%m-%d')
    with requests.Session() as session:
        data = apimoex.get_market_history(session, ticker, start_date, end_date)
        df = pd.DataFrame(data)
        df.set_index('TRADEDATE', inplace=True)
        return df[df['BOARDID']==boardid]
    
def get_index_prices_from_moex(ticker:str, boardid:str, market: str) -> pd.DataFrame:
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=3*365)).strftime('%Y-%m-%d')
    with requests.Session() as session:
        data = apimoex.get_board_history(session, 'IMOEX', board='SNDX',start=start_date,end=end_date, market='index')
        df = pd.DataFrame(data)
        df.set_index('TRADEDATE', inplace=True)
        return df

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
    description = data.iloc[0]['Name']
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

def calculate_market_value_of_debt(total_debt, interest_expense, r_d=0.05, n=10):
    """
    Estimates Market Value of Debt
    r_d: Cost of debt (market rate) from interest payments or bond yields 
    n: Maturity in years
    """
    if not total_debt or total_debt == 0:
        return 0
    
    # Calculate MVD using bond pricing formula
    mvd = (interest_expense * (1 - (1 + r_d)**-n) / r_d) + (total_debt / (1 + r_d)**n)
    return mvd

def calculate_capm(market:str,ticker:str, rf:float = 0.145, rm: float = 0.135) -> float:
    """   
    :param market: market - US market "NYSE", or RU market "MOEX", decides which index to take
    :type market: str
    :param ticker: ticker code "AAPL", "F", etc.
    :type ticker: str
    :param rf: risk free rate, 10Y treasuries bond yield (ca. 4% as of 2025), or long OFZ yield alson 10y (ca. 13% as of 2025) 
    :type rf: float
    :param rm: market rate - assumingly SP500  returns over 3 last years on average, IMOEX returns for RU market
    :type rm: float
    :return: expected returns
    :rtype: float
    """
    if market != 'MOEX':
        stock_prices = yf.download(tickers=ticker,period='3y')
        stock_close_prices = stock_prices['Close'][ticker]
        index_ticker = "^GSPC"
        index_price = yf.download(tickers=index_ticker,period='3y')
        index_close_price = index_price['Close'][index_ticker]
        stock_returns = np.log(stock_close_prices / stock_close_prices.shift(1))
        index_returns = np.log(index_close_price / index_close_price.shift(1))
    else:
        index_price = 'IMOEX'
        stock_prices = get_prices_from_moex(ticker,'TQBR', 'shares')#TQOB for bonds, TQBR for stocks
        stock_close_prices = stock_prices['CLOSE']
        index_price = get_index_prices_from_moex("IMOEX",'SNDX', 'index')#TQOB for bonds, TQBR for stocks
        index_close_price = index_price['CLOSE']
        stock_returns = np.log(stock_close_prices / stock_close_prices.shift(1))
        index_returns = np.log(index_close_price / index_close_price.shift(1))
    # Объединение данных
    returns = pd.concat([stock_returns, index_returns], axis=1).dropna()
    returns.columns = ['Stock', 'Index']
    # Линейная регрессия
    X = sm.add_constant(returns['Index'])
    model = sm.OLS(returns['Stock'], X).fit()
    beta = model.params['Index']
    # Расчёт по формуле CAPM
    expected_return = rf + beta * (rm - rf)
    return expected_return

def calculate_wacc(equity_value:float,debt_value:float,tax_rate:float,cost_of_equity:float,cost_of_debt:float) -> float:
    """
    :param equity_value: market capitalization, absolute value
    :type equity_value: float
    :param debt_value: market value of debt, use calculate_market_value_of_debt to get it.
    :type debt_value: float
    :param tax_rate: effective tax rate in country of calculation, i.e. 0.2 in russia in 2025
    :type tax_rate: float
    :param cost_of_equity: result of CAPM
    :type cost_of_equity: float
    :param cost_of_debt: bond yield of a given company
    :type cost_of_debt: float
    :return: WACC rate
    :rtype: float
    """
    total_capital = equity_value + debt_value
    wacc = (equity_value / total_capital) * cost_of_equity + \
        (debt_value / total_capital) * cost_of_debt * (1 - tax_rate)
    return wacc

def calculate_dcf_fcf(fcf:list[str],growth_rate:float,years:int,discount_rate:float,terminal_growth:float,net_debt:float,
                      shares_outstanding:float) -> float:
    """
    :param fcf: 3 years list of free cash flow
    :type fcf: list[str]
    :param growth_rate: growth rate, may be average net income growth rate
    :type growth_rate: float
    :param years: number of years to do projection for
    :type years: int
    :param discount_rate: WACC
    :type discount_rate: float
    :param terminal_growth: Long-term GDP-level growth (3%)
    :type terminal_growth: float
    :param net_debt: cash minus debt (negative if cash-rich) 
    :type net_debt: float
    :param shares_outstanding: number of shares outstanding
    :type shares_outstanding: float
    :return: fair value according to DCF based on FCF
    :rtype: float
    """
    last_fcf = fcf[-1]
    fcf_proj = [last_fcf * ((1 + growth_rate) ** i) for i in range(1, years + 1)]
    discount_factors = [(1 / (1 + discount_rate) ** i) for i in range(1, years + 1)]
    pv_fcf = [fcf_proj[i] * discount_factors[i] for i in range(years)]

    # Terminal value
    terminal_value = (fcf_proj[-1] * (1 + terminal_growth)) / (discount_rate - terminal_growth)
    pv_terminal = terminal_value / ((1 + discount_rate) ** years)

    # Total equity value
    enterprise_value = sum(pv_fcf) + pv_terminal
    equity_value = enterprise_value + net_debt
    fair_value_per_share = equity_value / shares_outstanding
    return fair_value_per_share

def get_interest_expense_from_file(ticker:str, exchange:str):
    if exchange == 'MOEX':
        df = pd.read_csv('moex_data.csv')
        interest_expense_row = df[df['Ticker'] == ticker]
        if len(interest_expense_row)>0:
            interest_expense = df[df['Ticker'] == ticker].iloc[0]['InterestExpense']
        else:
            interest_expense = None
        return float(interest_expense)
    else:
        df = pd.read_csv('net_income_nyse_nasdaq.csv')
        interest_expense_row = df[df['tickers'] == ticker]
        if len(interest_expense_row)>0:
            interest_expense = df[df['tickers'] == ticker].iloc[0]['interest_expense']
        else:
            interest_expense = None
        return float(interest_expense)

def get_dcf_valuation(exchange:str, ticker: str):
    if exchange == "MOEX":
        capm = calculate_capm(market=exchange, ticker=ticker, rm=0.10, rf=0.13)
    else:
        capm = calculate_capm(market=exchange, ticker=ticker, rm=0.10, rf=0.04)
    if exchange == 'MOEX':
        df = pd.read_csv('moex_data.csv')
        row = df[df['Ticker'] == ticker].iloc[0]
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
                        "debt_to_equity",
                        "total_debt",
                        "net_debt",
                        "total_shares_outstanding_fundamental",
                        "effective_interest_rate_on_debt_ttm"
                    )
                    .where(col("name") == ticker)
                    .get_scanner_data()
                )
        row = df[1].iloc[0]
    if exchange == 'MOEX':
        close = get_price_from_moex(ticker=ticker)
        equity_value = round(close * float(row['Issue']),2)
        total_debt = row['Debt']
        cost_of_debt = float(row['InterestRateOnDebt']) 
        fcf = [row["FCF"]] # seems like only last fcf is needed for calc
        net_debt =row['NetDebt']
        shares_outstanding = row['Issue']
    else:
        equity_value = row['market_cap_basic']
        total_debt = row['total_debt']
        cost_of_debt = float(row['effective_interest_rate_on_debt_ttm'])/100  # downloaded to file for moex
        fcf = [row["free_cash_flow_fy"]] # seems like only last fcf is needed for calc
        net_debt =row['net_debt']
        shares_outstanding = row['total_shares_outstanding_fundamental']
    interest_expense = get_interest_expense_from_file(ticker=ticker,exchange=exchange)
    debt_value = calculate_market_value_of_debt(total_debt=total_debt,
                                                 interest_expense=interest_expense,r_d=cost_of_debt,n=3)
    wacc = calculate_wacc(equity_value=equity_value,debt_value=debt_value,tax_rate=0.21,
                          cost_of_equity=capm,cost_of_debt=cost_of_debt)
    netProfitHistory = get_net_income_from_file(ticker, exchange)
    if netProfitHistory is not None and len(netProfitHistory)>=5:
        averageGrowth: AverageGrowth = calculate_average_growth(netProfitHistory)
    else:
        averageGrowth: AverageGrowth = None
    if averageGrowth is not None and averageGrowth.fiveYears is not None:
        growth_rate = averageGrowth.fiveYears
    else:
        growth_rate = 0
    if growth_rate > 25:
        growth_rate = 25
    elif growth_rate <0:
        growth_rate = 0
    else:
        growth_rate = growth_rate
    if exchange == "MOEX":
        terminal_growth = 0.043 # Russian GDP Growth Rate 2024
    else:
        terminal_growth = 0.028 # US GDP Growth Rate 2024
    dcf_fcf_fair_value = calculate_dcf_fcf(fcf=fcf,growth_rate=growth_rate/100, years=3,discount_rate=wacc,
                                           terminal_growth=terminal_growth,
                                           net_debt=net_debt,shares_outstanding=shares_outstanding)
    return dcf_fcf_fair_value
