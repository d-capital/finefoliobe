from models.exchange import ExchangeRate
import yfinance as yf
import datetime as dt
from datetime import datetime
import yfinance as yf
from alpha_vantage.foreignexchange import ForeignExchange
api_key = os.environ.get('ALPHA_VANTAGE_API_KEY') 
fx = ForeignExchange(key=api_key, output_format='pandas')

import os
import json
import pandas as pd


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def get_exchange_rate(ticker: str):
    first_currency = ticker.split(".")[0].lower()
    second_currency = ticker.split(".")[1].lower()
    if first_currency and second_currency:
        result = prepare_data_for_regression(first_currency, second_currency)
        generate_ppp(result)
        
        data=result

        # Step 3: Preprocessing
        data = data.dropna()  # Drop rows with missing values


        data['exchange_rate'] = data['exchange_rate'].diff(2)
        data['fcurr_ir'] = data['fcurr_ir'].diff(1)
        data['fcurr_inf'] = data['fcurr_inf'].diff(1)
        data['fcurr_unem'] = data['fcurr_unem'].diff(1)
        data['fcurr_ir'] = data['fcurr_ir'].diff(1)
        data['fcurr_inf'] = data['fcurr_inf'].diff(1)
        data['fcurr_unem'] = data['fcurr_unem'].diff(1)
        #data['fcurr_gdp'] = data['fcurr_gdp'].diff(1)

        data['scurr_ir'] = data['scurr_ir'].diff(1)
        data['scurr_inf'] = data['scurr_inf'].diff(1)
        data['scurr_unem'] = data['scurr_unem'].diff(1)
        data['scurr_ir'] = data['scurr_ir'].diff(1)
        data['scurr_inf'] = data['scurr_inf'].diff(1)
        data['scurr_unem'] = data['scurr_unem'].diff(1)
        #data['scurr_gdp'] = data['scurr_gdp'].diff(1)

        data = data.dropna()  # Drop rows again after shifting

        # Step 4: Feature and Target
        X = data.drop(columns=['date', 'exchange_rate'])
        y = data['exchange_rate']

        # Step 5: Split into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Step 6: Train the model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Step 7: Evaluate the model
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        print(f"Mean Absolute Error: {mae}")
        print(f"Mean Squared Error: {mse}")

        # Step 9: Feature Importance
        feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
        feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

        forecasted_regression_rate = round(float(result.iloc[-1]['exchange_rate']) + float(y_pred[-1]),5)
        current_rate = round(float(result.iloc[-1]['exchange_rate']),5)

        return ExchangeRate(
            rate = current_rate,
            first_currency_short_code=first_currency,
            first_currency_interest_rate = float(result.iloc[-1]['fcurr_ir']),
            first_currency_inflation_rate = float(result.iloc[-1]['fcurr_inf']),
            first_currency_unemployment_rate = float(result.iloc[-1]['fcurr_gdp']),
            first_currency_gdp_growth_rate = 0.0,
            second_currency_short_code = second_currency,
            second_currency_interest_rate = float(result.iloc[-1]['scurr_ir']),
            second_currency_inflation_rate = float(result.iloc[-1]['scurr_inf']),
            second_currency_unemployment_rate = float(result.iloc[-1]['scurr_unem']),
            second_currency_gdp_growth_rate = float(result.iloc[-1]['scurr_gdp']),
            forecast_regression = forecasted_regression_rate,
            forecast_ppp = round(float(result.iloc[-1]['ppp_avg']),5),
            recommendation = "sell" if current_rate > forecasted_regression_rate else "buy" 
        )
    return None


def get_interest_rates(country: str) -> list:
    path:str = os.getcwd() + "/interest_rate.json"
    with open(path, 'r') as j:
        data:list = json.load(j)
        interest_rates = data[country]
        df_ir = pd.DataFrame.from_records(interest_rates)
        return df_ir
    
def get_inflation_rates(country: str) -> list:
    path:str = os.getcwd() + "/inflation.json"
    with open(path, 'r') as j:
        data:list = json.load(j)
        interest_rates = data[country]
        df_ir = pd.DataFrame.from_records(interest_rates)
        return df_ir
    
def get_unemployment_rates(country: str) -> list:
    path:str = os.getcwd() + "/unemployment.json"
    with open(path, 'r') as j:
        data:list = json.load(j)
        interest_rates = data[country]
        df_ir = pd.DataFrame.from_records(interest_rates)
        return df_ir
    
def get_gdp_rates(country: str) -> list:
    path:str = os.getcwd() + "/gdp.json"
    with open(path, 'r') as j:
        data:list = json.load(j)
        interest_rates = data[country]
        df_ir = pd.DataFrame.from_records(interest_rates)
        return df_ir
    

def format_date_reg(df: pd.DataFrame):
  df['date'] = None
  for i in range(len(df)):
    new_date = datetime.utcfromtimestamp(int(df['dateline'].iloc[i]))
    formatted_date = new_date.strftime('%d.%m.%Y')
    df.at[i,'date'] = formatted_date

def prepare_data_for_regression(first_currency:str,second_currency:str):

    #FIRST CURRENCY
    #interest rate
    df_ir_f_currency = get_interest_rates(first_currency)
    format_date_reg(df_ir_f_currency)
    df_ir_f_currency = df_ir_f_currency.set_index('date')
    df_ir_f_currency_f = df_ir_f_currency[['actual']]
    df_ir_f_currency_f = df_ir_f_currency_f.rename(columns={'actual':'fcurr_ir'})

    #inflation rate
    df_inf_f_currency = get_inflation_rates(first_currency)
    format_date_reg(df_inf_f_currency)
    df_inf_f_currency = df_inf_f_currency.set_index('date')
    df_inf_f_currency_f = df_inf_f_currency[['actual']]
    df_inf_f_currency_f = df_inf_f_currency_f.rename(columns={'actual':'fcurr_inf'})

    #unemployment
    df_u_f_currency = get_unemployment_rates(first_currency)
    format_date_reg(df_u_f_currency)
    df_u_f_currency = df_u_f_currency.set_index('date')
    df_u_f_currency_f = df_u_f_currency[['actual']]
    df_u_f_currency_f = df_u_f_currency_f.rename(columns={'actual':'fcurr_unem'})

    #GDP
    df_gdp_f_currency = get_gdp_rates(first_currency)
    format_date_reg(df_gdp_f_currency)
    df_gdp_f_currency = df_gdp_f_currency.set_index('date')
    df_gdp_f_currency_f = df_gdp_f_currency[['actual']]
    df_gdp_f_currency_f = df_gdp_f_currency_f.rename(columns={'actual':'fcurr_gdp'})

    #SECOND CURRENCY
    #interest rate
    df_ir_s_currency = get_interest_rates(second_currency)
    format_date_reg(df_ir_s_currency)
    df_ir_s_currency = df_ir_s_currency.set_index('date')
    df_ir_s_currency_f = df_ir_s_currency[['actual']]
    df_ir_s_currency_f = df_ir_s_currency_f.rename(columns={'actual':'scurr_ir'})

    #inflation rate
    df_inf_s_currency = get_inflation_rates(second_currency)
    format_date_reg(df_inf_s_currency)
    df_inf_s_currency = df_inf_s_currency.set_index('date')
    df_inf_s_currency_f = df_inf_s_currency[['actual']]
    df_inf_s_currency_f = df_inf_s_currency_f.rename(columns={'actual':'scurr_inf'})

    #unemployment
    df_u_s_currency = get_unemployment_rates(second_currency)
    format_date_reg(df_u_s_currency)
    df_u_s_currency = df_u_s_currency.set_index('date')
    df_u_s_currency_f = df_u_s_currency[['actual']]
    df_u_s_currency_f = df_u_s_currency_f.rename(columns={'actual':'scurr_unem'})

    #GDP
    df_gdp_s_currency = get_gdp_rates(second_currency)
    format_date_reg(df_gdp_s_currency)
    df_gdp_s_currency = df_gdp_s_currency.set_index('date')
    df_gdp_s_currency_f = df_gdp_s_currency[['actual']]
    df_gdp_s_currency_f = df_gdp_s_currency_f.rename(columns={'actual':'scurr_gdp'})

    #FIRST CURRENCY MERGE
    df_ir_f_currency_f.index = pd.to_datetime(df_ir_f_currency_f.index,format="%d.%m.%Y")
    df_inf_f_currency_f.index = pd.to_datetime(df_inf_f_currency_f.index,format="%d.%m.%Y")
    df_u_f_currency_f.index = pd.to_datetime(df_u_f_currency_f.index,format="%d.%m.%Y")
    df_gdp_f_currency_f.index = pd.to_datetime(df_gdp_f_currency_f.index,format="%d.%m.%Y")

    ir_inf_f_currency = pd.merge_asof(df_ir_f_currency_f.sort_index(), df_inf_f_currency_f.sort_index(), left_index=True, right_index=True)
    ir_inf_f_currency.index = pd.to_datetime(ir_inf_f_currency.index,format="%d.%m.%Y")
    ir_inf_u_f_currency = pd.merge_asof(ir_inf_f_currency.sort_index(), df_u_f_currency_f.sort_index(), left_index=True, right_index=True)
    ir_inf_u_f_currency.index = pd.to_datetime(ir_inf_u_f_currency.index,format="%d.%m.%Y")
    ir_inf_u_gdp_f_currency = pd.merge_asof(ir_inf_u_f_currency.sort_index(), df_gdp_f_currency_f.sort_index(), left_index=True, right_index=True)

    #SECOND CURRENCY MERGE
    df_ir_s_currency_f.index = pd.to_datetime(df_ir_s_currency_f.index,format="%d.%m.%Y")
    df_inf_s_currency_f.index = pd.to_datetime(df_inf_s_currency_f.index,format="%d.%m.%Y")
    df_u_s_currency_f.index = pd.to_datetime(df_u_s_currency_f.index,format="%d.%m.%Y")
    df_gdp_s_currency_f.index = pd.to_datetime(df_gdp_s_currency_f.index,format="%d.%m.%Y")

    ir_inf_s_currency = pd.merge_asof(df_ir_s_currency_f.sort_index(), df_inf_s_currency_f.sort_index(), left_index=True, right_index=True)
    ir_inf_s_currency.index = pd.to_datetime(ir_inf_s_currency.index,format="%d.%m.%Y")
    ir_inf_u_s_currency = pd.merge_asof(ir_inf_s_currency.sort_index(), df_u_s_currency_f.sort_index(), left_index=True, right_index=True)
    ir_inf_u_s_currency.index = pd.to_datetime(ir_inf_u_s_currency.index,format="%d.%m.%Y")
    ir_inf_u_gdp_s_currency = pd.merge_asof(ir_inf_u_s_currency.sort_index(), df_gdp_s_currency_f.sort_index(), left_index=True, right_index=True)

    #MERGE BOTH CURRENCIES
    econ_data = pd.merge_asof(ir_inf_u_gdp_f_currency.sort_index(), ir_inf_u_gdp_s_currency.sort_index(), left_index=True, right_index=True)

    #LOAD EXCHANGE RATE
    ticker = first_currency.upper() + second_currency.upper() + '=X'
    #rate = yf.download(ticker,start='2000-03-22',end='2025-05-10')
    
    rate, meta_data = fx.get_currency_exchange_daily(from_symbol=first_currency.upper(), to_symbol=second_currency.upper(), outputsize='full')
    #rate = rate['Close'][ticker].to_frame()
    rate = rate.rename(columns={'4. close':'exchange_rate'})
    data = pd.merge_asof(econ_data.sort_index(), rate.sort_index(),left_index=True,right_index=True)

    data = data.reset_index()
    
    return data

#PPP
def generate_ppp(df:pd.DataFrame):
    df['ppp'] = None
    df['ppp_avg'] = None
    for i in range(len(df)):
        fc_inf = df.at[i,'fcurr_inf']/100
        sc_inf = df.at[i,'scurr_inf']/100
        ex = df.at[i,'exchange_rate']
        df.at[i,'ppp'] = ex * ((1 + fc_inf) / (1 + sc_inf))
    df['ppp_avg'] = df['ppp'].mean()
