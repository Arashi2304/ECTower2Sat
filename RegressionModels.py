import numpy as np
import random
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score

def prep_data(data_dict, split=0.05):
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    
    for key, df in data_dict.items():
        for index, row in df.iterrows():
            y_train.append(row['NEE'])
            #sr = [row['ST_B6']]
            sr = [row['SR_B1'], row['SR_B2'], row['SR_B3'], row['SR_B4'], row['SR_B5'], row['SR_B7'], row['ST_B6']]
            #sr = [row['SR_B1'], row['SR_B2'], row['SR_B3'], row['SR_B4'], row['SR_B5'], row['SR_B7'], row['B1'], row['B2'], row['B3'], row['B4'], row['B5'], row['B7']]
            #sr = [row['B1'], row['B2'], row['B3'], row['B4'], row['B5'], row['B7']]
            #sr = [row['D1'], row['D2'], row['D3'], row['D4'], row['D5'], row['D7']]
            x_train.append(sr)
    
    y_train = pd.to_numeric(y_train, errors='coerce')
    
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    
    random.shuffle(x_train)
    random.shuffle(y_train)
    
    index = int(len(x_train) * (1 - split))
    x_test = x_train[index:]
    y_test = y_train[index:]
    x_train = x_train[:index]
    y_train = y_train[:index]
    
    return x_train, x_test, y_train, y_test

class RegressionModels:
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        
        self.models = {}
        self.scores = {}
        self.errors = {}

        # Train models
        self.train_random_forest()
        self.train_linear_regression()
        self.train_ridge_regression()
        self.train_lasso_regression()
        self.train_elastic_net()
        self.train_svr()
        self.train_gradient_boosting()
        self.train_ann()
        
    def train_random_forest(self):
        rf = RandomForestRegressor()
        rf.fit(self.x_train, self.y_train)
        score = rf.score(self.x_test, self.y_test)
        self.models['RandomForest'] = rf
        self.scores['RandomForest'] = score

    def train_linear_regression(self):
        lr = LinearRegression()
        lr.fit(self.x_train, self.y_train)
        score = lr.score(self.x_test, self.y_test)
        self.models['LinearRegression'] = lr
        self.scores['LinearRegression'] = score

    def train_ridge_regression(self):
        ridge = Ridge()
        ridge.fit(self.x_train, self.y_train)
        score = ridge.score(self.x_test, self.y_test)
        self.models['RidgeRegression'] = ridge
        self.scores['RidgeRegression'] = score

    def train_lasso_regression(self):
        lasso = Lasso()
        lasso.fit(self.x_train, self.y_train)
        score = lasso.score(self.x_test, self.y_test)
        self.models['LassoRegression'] = lasso
        self.scores['LassoRegression'] = score

    def train_elastic_net(self):
        en = ElasticNet()
        en.fit(self.x_train, self.y_train)
        score = en.score(self.x_test, self.y_test)
        self.models['ElasticNet'] = en
        self.scores['ElasticNet'] = score

    def train_svr(self):
        svr = SVR()
        svr.fit(self.x_train, self.y_train)
        score = svr.score(self.x_test, self.y_test)
        self.models['SVR'] = svr
        self.scores['SVR'] = score

    def train_gradient_boosting(self):
        gb = GradientBoostingRegressor()
        gb.fit(self.x_train, self.y_train)
        score = gb.score(self.x_test, self.y_test)
        self.models['GradientBoosting'] = gb
        self.scores['GradientBoosting'] = score

    def train_ann(self):
        ann = Sequential()
        ann.add(Dense(8, input_dim=self.x_train.shape[1], activation='relu'))
        ann.add(Dense(4, activation='relu'))
        ann.add(Dense(1))
        
        ann.compile(loss='mean_squared_error', optimizer='adam')
        ann.fit(self.x_train, self.y_train, epochs=100, batch_size=10, verbose=0)
        
        y_pred = ann.predict(self.x_test)
        score = r2_score(self.y_test, y_pred)
        
        self.models['ANN'] = ann
        self.scores['ANN'] = score