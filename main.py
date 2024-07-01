from data import combined_data
from RegressionModels import RegressionModels, prep_data
import pandas as pd

data_dict_LST, data_dict_MDS = combined_data()
#data_dict = combined_data(Towers=['K67'])
x_train, x_test, y_train, y_test = prep_data(data_dict_MDS)

Run = RegressionModels(x_train, y_train, x_test, y_test)
print(Run.scores)