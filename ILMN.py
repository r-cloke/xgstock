# -*- coding: utf-8 -*-
"""
Created on Sun May 12 17:46:16 2019

@author: rcloke
"""

import pandas_datareader as pdr
import datetime 
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

name='AMZN'
stock = pdr.get_data_yahoo(name, 
                          start=datetime.datetime(2006, 10, 1), 
                          end=datetime.datetime(2019, 4, 1))


# Plot the closing prices for `aapl`
stock['Close'].plot(grid=True)

# Show the plot
plt.show()

train = stock

for i in range(1,10):
    train['t-'+str(i)] = train['Close'].shift(i)
for i in range(1,10):
    train['vt-'+str(i)] = train['Volume'].shift(i)
    
model_input = train
print(model_input.head())

X, y = model_input.iloc[:,np.r_[6:24]],model_input.iloc[:,3]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

xg_reg = XGBRegressor(max_depth=10, learning_rate=0.1, n_estimators=1000, 
                      silent=True, objective='reg:linear', 
                      nthread=None, gamma=0, min_child_weight=1, max_delta_step=0, subsample=1, 
                      colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, 
                      base_score=0.5, missing=None)

xg_reg.fit(X_train,y_train)
plt.rcParams['figure.figsize'] = [50, 10]
plt.show()
xgb.plot_importance(xg_reg,max_num_features=3)
plt.rcParams['figure.figsize'] = [5, 5]
plt.show()

preds = xg_reg.predict(X_test)
predictions = np.ndarray.reshape(preds,(preds.shape[0],1))
plt.plot(y_test,predictions,'ro')
plt.show

rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))

modelName =name+'_XGB.model'
xg_reg.save_model(modelName)