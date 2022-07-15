import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import pickle



train = pd.read_excel ('train_set_X.xlsx', sheet_name='Sheet1' ,engine='openpyxl')
test = pd.read_excel ('test_set.xlsx', sheet_name='Sheet1' ,engine='openpyxl')

test= test.drop(['CONTRACT_STATUS','PAID_RENTALS','Reschedule','CStatus','NET_RENTAL_NEW'], axis =1)
train= train.drop(['CONTRACT_STATUS', 'PAID_RENTALS','Reschedule','CStatus','NET_RENTAL_NEW'], axis =1)

X_train= train.drop(['IS_DEFAULT'], axis=1)
X_test = test.drop(['IS_DEFAULT'], axis=1)
y_train = train['IS_DEFAULT']
y_test = test['IS_DEFAULT']



#xgb sklearn model
classfication_model = xgb.XGBClassifier(scale_pos_weight =13.49,
                                        min_child_weight= 5,
                                        max_depth= 4,
                                        learning_rate= 0.14,
                                        gamma= 0.3,
                                        colsample_bytree= 0.3, 
                                        n_estimators=100)
classfication_model.fit(X_train, y_train)
default_pred = classfication_model.predict(X_test)
default_pred_prob = classfication_model.predict_proba(X_test)



# Saving model to disk
pickle.dump(classfication_model, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
#print(model.predict([[8914, 36, 1982,250000,17,41,43000, 8000,1,7,13,0,16,0,1]]))