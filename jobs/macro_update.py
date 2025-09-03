from repositories.macrodata import MacroDataRepository
import investpy
from datetime import datetime as dt
from models.macro_event import MacroEvent
from sqlmodel import Session
from fastapi import Depends
from db.session import engine
import math
import pandas as pd
countries = ['new zealand', 'united kingdom', 'switzerland', 'euro zone', 
             'united states', 'canada', 'japan', 'australia']
eventlist = ['CPI (YoY)','Unemployment Rate','GDP (YoY)', 'Interest Rate', 'Trade Balance']
rate_change_event_list = ['ECB Interest Rate Decision','BoE Interest Rate Decision','Fed Interest Rate Decision',
                          'BoC Interest Rate Decision','BoJ Interest Rate Decision','RBA Interest Rate Decision',
                          'RBNZ Interest Rate Decision', 'SNB Interest Rate Decision']

event_dict = {
    'CPI (YoY)':'inf',
    'Unemployment Rate':'une',
    'GDP (YoY)':'gdp',
    'Interest Rate':'ir',
    'Trade Balance':'tb'
}

def update():
    #getdata
    #get date of last event saved, this will be `from_date`
    session: Session = Session(engine)
    last_event = MacroDataRepository(session=session).get_last_event()
    last_date = dt.strftime(dt.fromtimestamp(last_event.dateline),'%d/%m/%Y')
    #todays date will be `to_date`
    today = dt.today()
    str_today = dt.strftime(today,'%d/%m/%Y')
    #TODO: alog is not catching last ECB rate change, need to understand why
    data: pd.DataFrame = investpy.economic_calendar(countries = countries, from_date = last_date, to_date = str_today)
    #save data
    #TODO: it might be a problem to find correct events so I need to modify them in the df itself before filtering
    for i in range(len(data)):
        el = data.iloc[i]
        event_name: str = str(el['event'])
        if event_name.startswith('Unemployment Rate'):
            data.at[i, 'event'] = 'Unemployment Rate'
        if event_name.startswith('CPI (YoY)'):
            data.at[i, 'event'] = 'CPI (YoY)'
        if event_name.startswith('GDP (YoY)'):
            data.at[i, 'event'] = 'GDP (YoY)'
        if is_rate_change_event(event_name):
            data.at[i, 'event'] = 'Interest Rate'
        if event_name.startswith('Trade Balance (YoY)'):
            data.at[i, 'event'] = 'Trade Balance'
    #filter out events that match events that we need by country and even name
    needed_events = data[data['event'].isin(eventlist)]
    needed_events = needed_events.dropna(subset=['actual'])
    #for each of such events 
    for i in range(len(needed_events)):
        event = needed_events.iloc[i]
        print(str(event['currency']).lower())
        print(event['date'])
        print(event['forecast'])
        event_to_save = MacroEvent(
            country=str(event['currency']).lower(),
            event_type=event_dict[event['event']],
            dateline=get_dateline(event),
            date=event['date'],
            actual=format_number(event['actual']),
            actual_formatted=event['actual'],
            forecast=get_formatted_event(event['forecast']),
            forecast_formatted=event['forecast'],
            is_active=True,
            is_most_recent=True
        )
        last_event: MacroEvent = MacroDataRepository(session=session).get_last_event_by_type_and_country(country=event_to_save.country,type=event_to_save.event_type)
        if(event_to_save.dateline != last_event.dateline):
            repo  = MacroDataRepository(session=session).create_new(macroEvent=event_to_save)
    print('Finished updates')

def is_rate_change_event(event_name: str) -> bool:
    number_of_matches = 0
    for i in rate_change_event_list:
        if event_name.startswith(i):
            number_of_matches += 1
    return number_of_matches > 0

def get_dateline(df:pd.DataFrame) -> int:
    time = '00:00'
    if df['time'] == '' or df['time'] is None:
        time = '00:00'
    else:
        time = df['time']
    date_str = df['date'] + ' ' + time
    # Convert to datetime object (specify format for correct parsing)
    datetime_obj = dt.strptime(date_str, '%d/%m/%Y %H:%M')
    # Convert to UNIX timestamp (seconds since epoch)
    dateline = int(datetime_obj.timestamp())
    return dateline

def format_date(date_str:str) -> str:
    date_obj = dt.strptime(date_str, '%m/%d/%Y')
    formatted_date = date_obj.strftime('%b %Y')
    return formatted_date

def format_number(number:str):
    formatted_number = number
    formatted_number = formatted_number.replace(',','')
    if formatted_number.endswith('%'):
        formatted_number = float(formatted_number.strip('%'))
    elif formatted_number.endswith('B'):
        formatted_number = float(formatted_number.strip('B'))
    elif formatted_number.endswith('M'):
        formatted_number = formatted_number.strip('M')
        formatted_number = float(formatted_number)/1000
    return formatted_number


def get_trade_balance():
    session: Session = Session(engine)
    last_event = MacroDataRepository(session=session).get_last_event()
    last_date = dt.strftime(dt.fromtimestamp(last_event.dateline),'%d/%m/%Y')
    #todays date will be `to_date`
    # today = dt.today()
    # str_today = dt.strftime(today,'%d/%m/%Y')
    # #TODO: alog is not catching last ECB rate change, need to understand why
    # data: pd.DataFrame = investpy.economic_calendar(countries = countries, categories=['balance'], from_date = '01/01/2003', to_date = str_today)
    # #save data
    # #TODO: it might be a problem to find correct events so I need to modify them in the df itself before filtering
    # for i in range(len(data)):
    #     el = data.iloc[i]
    #     event_name: str = str(el['event'])
    #     if event_name.startswith('Trade Balance'):
    #         data.at[i, 'event'] = 'Trade Balance'
    # #filter out events that match events that we need by country and even name
    # needed_events = data[data['event'].isin(eventlist)]
    # needed_events = needed_events.dropna(subset=['actual'])
    #for each of such events 
    needed_events = pd.read_csv('trade_balances_2.csv')
    for i in range(len(needed_events)):
        event = needed_events.iloc[i]
        print(str(event['currency']).lower())
        print(event['date'])
        print(event['forecast'])
        event_to_save = MacroEvent(
            country=str(event['currency']).lower(),
            event_type=event_dict[event['event']],
            dateline=get_dateline(event),
            date=event['date'],
            actual=format_number(event['actual']),
            actual_formatted=event['actual'],
            forecast=get_formatted_event(event['forecast']),
            forecast_formatted=event['forecast'],
            is_active=True,
            is_most_recent=True
        )
        #last_event: MacroEvent = MacroDataRepository(session=session).get_last_event_by_type_and_country(country=event_to_save.country,type=event_to_save.event_type)
        #if(event_to_save.dateline != last_event.dateline):
        repo  = MacroDataRepository(session=session).create_new(macroEvent=event_to_save)
    print('Finished updates')

def get_formatted_event(value):
    print(value)
    print(type(value))
    if value is None:
        return None
    if type(value) != str:
        if math.isnan(value) == False:
            return None
    else:
        return format_number(value)