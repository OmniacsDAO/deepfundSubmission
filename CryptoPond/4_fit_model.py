import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

## Load Data
trainfeature_df = pd.read_csv('CryptoPondData/trainfeatures.csv')
testfeature_df = pd.read_csv('CryptoPondData/testfeatures.csv')
testd = pd.read_csv('CryptoPondData/test.csv')

## Train test Split
X = trainfeature_df.iloc[:, :-1]
y = trainfeature_df.Y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.075, random_state=12)

## XGBoost Parameter Search
# xgb1 = XGBRegressor()
# parameters = {
#               'objective':['reg:squarederror'],
#               'eval_metric':['rmse'],
#               'learning_rate': [0.01, .05, 0.1],
#               'max_depth': [4, 5, 6],
#               'verbosity': [1],
#               'subsample': [0.7,1],
#               'colsample_bytree': [0.5,0.7],
#               'n_estimators': [100, 250, 500]
#             }
# xgb_grid = GridSearchCV(xgb1,parameters,n_jobs = 1,verbose=True,cv=5)
# xgb_grid.fit(X_train, y_train)
# print(xgb_grid.best_score_)
# print(xgb_grid.best_params_)
regr = XGBRegressor(objective ='reg:squarederror',eval_metric = 'rmse', n_estimators = 250, max_depth=6,learning_rate=0.05, seed = 12, subsample = 1, colsample_bytree=.5)
regr.fit(X_train, y_train) 
# regr = xgb_grid.best_estimator_
print(mean_squared_error(y_test, regr.predict(X_test)))
print(mean_squared_error(y_train, regr.predict(X_train)))
submission = testd[['id']]
submission['pred'] = regr.predict(testfeature_df)
submission.to_csv('CryptoPondData/subxgb.csv',index=False)