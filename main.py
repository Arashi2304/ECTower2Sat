from data import combined_data
from RegressionModels import RegressionModels, prep_data
import pandas as pd

data_dict = combined_data()
x_train, x_test, y_train, y_test = prep_data(data_dict)

Run = RegressionModels(x_train, y_train, x_test, y_test)
print(Run.scores)
