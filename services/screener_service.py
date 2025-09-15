from models.sreener_result import ScreenerResult
from models.screener_filter import ScreenerFilter
import pandas as pd
from tradingview_screener import Query, col
from services.valuation_service import get_valuation, get_net_income_from_file, calculate_average_growth
from models.valuation_result import ValuationResult, AverageGrowth

def get_screener(screenerFilters:ScreenerFilter) -> list[ScreenerResult]:
    s = (Query()
        .select('name', 
                'description',
                'exchange', 
                'sector',
                'industry',
                'country',
                'market_cap_basic',
                'price_earnings_ttm',
                'earnings_per_share_basic_ttm',
                'price_earnings_forward_fy',
                'eps_diluted_growth_percent_fy',
                'dividends_yield',
                'price_earnings_growth_ttm',
                'debt_to_equity',
                'free_cash_flow',
                'earnings_per_share_forecast_next_fq',
                'close')
        .where(
            col('market_cap_basic').between(0, 5_000_000_000),
            col('price_earnings_ttm') <= 25,
            col('eps_diluted_growth_percent_fy')>15,
            col('dividends_yield')>=0,
            col('price_earnings_growth_ttm')<1.2,
            col('debt_to_equity')<0.5,
            col('free_cash_flow')>0,
            #col('price_earnings_forward_fy')<15,
            col('exchange').isin(['NYSE','NASDAQ']),
        )
        .get_scanner_data())
    screener_results:list[ScreenerResult] = []
    new_df = s[1]
    for i in range(len(new_df)):
        el = new_df.iloc[i]
        net_income_history = get_net_income_from_file(el['name'], el['exchange'])
        valuation_result = get_short_valuation_results(net_income_history, el['earnings_per_share_basic_ttm'], el['close'])
        if valuation_result[0] is not None: #avgGrowth
            new_result = ScreenerResult(
                id=i+1,
                company=el['description'],
                ticker=el['name'],
                result= valuation_result[2], #result of valuation - overvalued / undervalued + percentage
                fairPrice=valuation_result[1], #fair value price
                pegTtm=el['price_earnings_growth_ttm'],
                netIncomeGrowthTtm=valuation_result[0].ttm,
                netIncomeGrowth3y=valuation_result[0].threeYears,
                netIncomeGrowth5y=valuation_result[0].fiveYears,
                netIncomeGrowthNext1y=valuation_result[0].threeYears,
                netIncomeGrowthNext3y=valuation_result[0].fiveYears,
                freeCashFlow=el['free_cash_flow'],
                deFy=el['debt_to_equity'],
                dividendYield=el['dividends_yield'],
                peTtm=el['price_earnings_ttm'],
                forwardPe=el['price_earnings_forward_fy'],
                price=el['close'],
                type=get_lynch_company_type(valuation_result[0]), 
                marketCap=el['market_cap_basic'],
                sector=el['sector'],
                industry=el['industry'],
                country=el['country']
            )
        else:
            new_result = ScreenerResult(
                id=i+1,
                company=el['description'],
                ticker=el['name'],
                result= None, #result of valuation - overvalued / undervalued + percentage
                fairPrice=None, #fair value price
                pegTtm=el['price_earnings_growth_ttm'],
                netIncomeGrowthTtm=None,
                netIncomeGrowth3y=None,
                netIncomeGrowth5y=None,
                netIncomeGrowthNext1y=None,
                netIncomeGrowthNext3y=None,
                freeCashFlow=el['free_cash_flow'],
                deFy=el['debt_to_equity'],
                dividendYield=el['dividends_yield'],
                peTtm=el['price_earnings_ttm'],
                forwardPe=el['price_earnings_forward_fy'],
                price=el['close'],
                type=None,
                marketCap=el['market_cap_basic'],
                sector=el['sector'],
                industry=el['industry'],
                country=el['country']
            )
        screener_results.append(new_result)

    return screener_results

def get_lynch_company_type(avgGrowth:AverageGrowth)->str:
    avgFiveYearsNetIncomeGrowth = avgGrowth.fiveYears
    lynch_company_type=""
    if avgFiveYearsNetIncomeGrowth is not None:
        if avgFiveYearsNetIncomeGrowth > 0 and avgFiveYearsNetIncomeGrowth <=2:
            lynch_company_type = 'Slow Grower'
        elif avgFiveYearsNetIncomeGrowth > 9 and avgFiveYearsNetIncomeGrowth <=12:
            lynch_company_type = 'Stalwart'
        elif avgFiveYearsNetIncomeGrowth > 20:
            lynch_company_type = 'Fast Grower'
    return lynch_company_type

def get_short_valuation_results(netProfitHistory, epsTtm, price):
    if netProfitHistory is not None and len(netProfitHistory)>=5:
        averageGrowth: AverageGrowth = calculate_average_growth(netProfitHistory)
    else:
        averageGrowth: AverageGrowth = None
    if averageGrowth is not None and averageGrowth.fiveYears is not None and epsTtm is not None:
        fairPrice = averageGrowth.fiveYears* epsTtm
    else:
        fairPrice = None
    if averageGrowth is not None and averageGrowth.fiveYears is not None and epsTtm is not None:
        resultPercent = round((fairPrice/price)-1,2)*100
    else:
        resultPercent = 0.0
    resultLabel = "Overvalued"
    if(resultPercent>0):
        resultLabel = "Undervalued"
    else:
        resultLabel = "Overvalued"
    return averageGrowth, fairPrice, resultLabel, resultPercent