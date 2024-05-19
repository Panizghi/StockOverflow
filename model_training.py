import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score

# Load preprocessed data
data_df = pd.read_csv("./input/creditcard.csv")

# Define target and predictors
target = 'Class'
predictors = [
    'Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
    'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
    'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28',
    'Amount'
]

# Train/test split
train_df, test_df = train_test_split(data_df, test_size=0.2, random_state=2018, shuffle=True)
train_df, valid_df = train_test_split(train_df, test_size=0.2, random_state=2018, shuffle=True)

# Prepare the train and valid datasets
dtrain = xgb.DMatrix(train_df[predictors], train_df[target].values)
dvalid = xgb.DMatrix(valid_df[predictors], valid_df[target].values)
dtest = xgb.DMatrix(test_df[predictors], test_df[target].values)

# Define parameters and watchlist
params = {
    'objective': 'binary:logistic',
    'eta': 0.039,
    'silent': True,
    'max_depth': 2,
    'subsample': 0.8,
    'colsample_bytree': 0.9,
    'eval_metric': 'auc',
    'random_state': 2018
}
watchlist = [(dtrain, 'train'), (dvalid, 'valid')]

# Train the model
model = xgb.train(params, dtrain, 1000, watchlist, early_stopping_rounds=50, maximize=True, verbose_eval=50)

# Feature importance plot
fig, (ax) = plt.subplots(ncols=1, figsize=(8, 5))
xgb.plot_importance(model, height=0.8, title="Features importance (XGBoost)", ax=ax, color="green")
plt.show()
fig.savefig('fraud-graphs/feature_importance.png', transparent=True, bbox_inches='tight')

# Predictions and evaluation
preds = model.predict(dtest)
print("ROC AUC score:", roc_auc_score(test_df[target].values, preds))

# Save the model
model.save_model('fraud_models/xgb_model.json')
