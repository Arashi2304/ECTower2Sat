import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score

def prep_data(data_dict):
    x_train = []
    y_train = []
    for key, df in data_dict.items():
        for index, row in df.iterrows():
            y_train.append(row['NEE'])
            sr = [row['SR_B1'], row['SR_B2'], row['SR_B3'], row['SR_B4'], row['SR_B5'], row['SR_B7']]
            x_train.append(sr)
    
    return x_train, y_train

class RegressionModels:
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        
        self.models = {}
        self.scores = {}

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
        ann.add(Dense(64, input_dim=self.x_train.shape[1], activation='relu'))
        ann.add(Dense(32, activation='relu'))
        ann.add(Dense(1))
        
        ann.compile(loss='mean_squared_error', optimizer='adam')
        ann.fit(self.x_train, self.y_train, epochs=100, batch_size=10, verbose=0)
        
        y_pred = ann.predict(self.x_test)
        score = r2_score(self.y_test, y_pred)
        
        self.models['ANN'] = ann
        self.scores['ANN'] = score
        
    def get_scores(self):
        return self.scores