from models.macro_event import MacroEvent
import os
import json

macro_data_sources: dict = {'interestrate':'/interest_rate.json',
                            'inflation':'/inflation.json',
                            'unemployment':'/unemployment.json',
                            'gdp':'/gdp.json'}

def get_event_data(event:str, country: str) -> list[MacroEvent]:
    event_file: str = macro_data_sources[event]
    path:str = os.getcwd() + event_file
    with open(path, 'r') as j:
        data:list = json.load(j)
        json_data = data[country]
        result = []
        for item in json_data:
            event = MacroEvent(
                id=item['id'],
                dateline=item['dateline'],
                date=item['date'],
                actual_formatted=item['actual_formatted'],
                actual=item['actual'],
                forecast_formatted=item['forecast_formatted'],
                forecast=item['forecast'],
                revision_formatted="",
                revision = 0.0,
                is_active=item['is_active'],
                is_most_recent=item['is_most_recent']
            )
            result.append(event)
        return result