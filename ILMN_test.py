# -*- coding: utf-8 -*-
"""
Created on Sun May 12 18:18:01 2019

@author: rcloke
"""
import pandas_datareader as pdr
import pandas as pd
import datetime 
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error
from math import sqrt
import dateutil.relativedelta

def modelTest(name):
    modelName = name+'_XGB.model'
    today = datetime.datetime.today()
    lastMonth = today - dateutil.relativedelta.relativedelta(months=1)
    
    stock = pdr.get_data_yahoo(name, 
                              start=lastMonth,end=today)
    bst = xgb.Booster({'nthread': 4})  # init model
    bst.load_model(modelName)  # load data

    for i in range(1,10):
        stock['t-'+str(i)] = stock['Close'].shift(i)
    for i in range(1,10):
        stock['vt-'+str(i)] = stock['Volume'].shift(i)
    
    train = stock.drop(['High', 'Low','Open','Adj Close','Volume','Close'], axis=1)
    dtest = xgb.DMatrix(train)
    ypred = bst.predict(dtest)
    stock['pred']=ypred
    
    #stock['Close'].plot(grid=True,label='actual close')
    #stock['pred'].plot(grid=True,label='prediction')
    rms = sqrt(mean_squared_error(stock['Close'], stock['pred']))
    #print('rms is ',rms)
    #print('actual close is ',stock['Close'][-1:])
    #print('prediced close is ', stock['pred'][-1:])
    #plt.legend(loc='upper left')
    #plt.show()
    
    stock.to_csv('test.csv')
        # dump model
    #bst.dump_model('dump.raw.txt')

    #add row
    newRow = pd.DataFrame({"t-1":stock['Close'],"t-2":stock['t-1'],"t-3":stock['t-2'],"t-4":stock['t-3'],"t-5":stock['t-4'],"t-6":stock['t-5'],"t-7":stock['t-6'],"t-8":stock['t-7'],"t-9":stock['t-8'],"vt-1":stock['Volume'],"vt-2":stock['vt-1'],"vt-3":stock['vt-2'],"vt-4":stock['vt-3'],"vt-5":stock['vt-4'],"vt-6":stock['vt-5'],"vt-7":stock['vt-6'],"vt-8":stock['vt-7'],"vt-9":stock['vt-8']})
    newRow = newRow.iloc[[-1]]
    newRow.to_csv('newrow.csv')
    
    dtest = xgb.DMatrix(newRow)
    ypred = bst.predict(dtest)
    newRow['pred']=ypred
    
    tomorrow_price = newRow['pred'][-1:].to_string().split(' ')[4]
    print('tomorrows predicted close is ', tomorrow_price[:6])
    return stock[9:], str(tomorrow_price[:6])

modelTest('ILMN')
