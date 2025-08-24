from pydantic import BaseModel
from typing import Optional


class Country(BaseModel):
    code: Optional[str]
    gdp_growth: Optional[float]
    gdp_growth_url: Optional[str]
    interest_rate: Optional[float]
    interest_rate_url: Optional[str]
    inflation_rate: Optional[float]
    inflation_rate_url: Optional[str]
    unemployment_rate: Optional[float]
    unemployment_rate_url: Optional[str]

class CountriesRequest(BaseModel):
    codes: list[str]