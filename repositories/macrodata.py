from sqlmodel import Session, select, desc
from models.macro_event import MacroEvent
from models.country import Country
class MacroDataRepository:
    def __init__(self, session: Session):
        self.session = session
        
    def get_all(self):
        return self.session.exec(select(MacroEvent)).all()
    def get_event_by_type_and_country(self, type:str, country: str):
        return self.session.exec(select(MacroEvent).where(MacroEvent.event_type == type, MacroEvent.country == country)).all()
    def get_last_event_by_type_and_country(self, type:str, country: str):
        return self.session.exec(select(MacroEvent).where(MacroEvent.event_type == type, MacroEvent.country == country)
                                 .order_by(desc(MacroEvent.dateline))).first()
    def create_new(self, macroEvent: MacroEvent):
        event = self.session.add(macroEvent)
        self.session.commit()
        self.session.close()
        return event
    def get_last_event(self) -> MacroEvent:
        return self.session.exec(select(MacroEvent).order_by(desc(MacroEvent.dateline))).first()
    def get_countries_last_events(self, countries:list[str]) -> list[Country]:
        countries_to_return: list[Country] = []
        for country in countries:
            gdp_growth: MacroEvent = self.get_last_event_by_type_and_country('gdp',country=country)
            inflation_rate: MacroEvent = self.get_last_event_by_type_and_country('inf',country=country)
            interest_rate: MacroEvent = self.get_last_event_by_type_and_country('ir',country=country)
            unemployment_rate: MacroEvent = self.get_last_event_by_type_and_country('une',country=country)
            new_country = Country(
                code=country,
                gdp_growth=gdp_growth.actual,
                gdp_growth_url='/macrodata/gdp/'+country,
                interest_rate=interest_rate.actual,
                interest_rate_url= '/macrodata/interestrate/'+country,
                inflation_rate= inflation_rate.actual,
                inflation_rate_url='/macrodata/inflation/'+country,
                unemployment_rate=unemployment_rate.actual,
                unemployment_rate_url='/macrodata/unemployment/'+country
                )
            countries_to_return.append(new_country)
        return countries_to_return