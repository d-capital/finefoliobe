from pydantic import BaseModel

class ExchangeRate(BaseModel):
    rate: float
    first_currency_short_code: str
    first_currency_interest_rate: float
    first_currency_inflation_rate: float
    first_currency_unemployment_rate: float
    first_currency_gdp_growth_rate: float
    second_currency_short_code: str
    second_currency_interest_rate: float
    second_currency_inflation_rate: float
    second_currency_unemployment_rate: float
    second_currency_gdp_growth_rate: float
    forecast_regression: float
    forecast_ppp: float
    recommendation: str