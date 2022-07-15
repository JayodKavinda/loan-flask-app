import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error as MSE
import pickle

df = pd.read_excel ('all_defaults_preprocessed.xlsx', sheet_name='Sheet1')

X_train, X_test, y_train, y_test = train_test_split(df.drop('arrears_rate',axis=1),df['arrears_rate'],test_size=0.30, random_state=101)


regressor = SVR(kernel = 'rbf')
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
rounded_pred = [np.round(x,2) for x in y_pred]

rmse = np.sqrt(MSE(y_test, rounded_pred))
print("RMSE : % f" %(rmse))

# Saving model to disk
pickle.dump(regressor, open('model2.pkl','wb'))