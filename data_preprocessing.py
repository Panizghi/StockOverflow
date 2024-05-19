import pandas as pd
import numpy as np
import os

# Set path based on environment
IS_LOCAL = False
PATH = "./input/credit-card-fraud-detection" if IS_LOCAL else "./input"
print(os.listdir(PATH))

# Load data
data_df = pd.read_csv(PATH + "/creditcard.csv")
print("Credit Card Fraud Detection data - rows:", data_df.shape[0], "columns:", data_df.shape[1])

# Display data information
print(data_df.head())
print(data_df.describe())

# Check for missing values
total = data_df.isnull().sum().sort_values(ascending=False)
percent = (data_df.isnull().sum() / data_df.isnull().count() * 100).sort_values(ascending=False)
print(pd.concat([total, percent], axis=1, keys=['Total', 'Percent']).transpose())
