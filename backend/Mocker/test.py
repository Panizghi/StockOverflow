import pandas as pd
import numpy as np

def generate_mock_data(num_non_fraud=284315, num_fraud=492):
    # Define the column names
    columns = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount', 'Class']
    
    # Generate non-fraud data
    non_fraud_data = {
        'Time': np.random.normal(loc=94813.859575, scale=47488.145955, size=num_non_fraud),
        'Amount': np.random.normal(loc=88.291022, scale=250.105092, size=num_non_fraud),
        'Class': np.zeros(num_non_fraud, dtype=int)
    }
    for i in range(1, 29):
        non_fraud_data[f'V{i}'] = np.random.normal(loc=0, scale=1, size=num_non_fraud)
    
    # Generate fraud data
    fraud_data = {
        'Time': np.random.normal(loc=94813.859575, scale=47488.145955, size=num_fraud),
        'Amount': np.random.normal(loc=122.211321, scale=256.683288, size=num_fraud),
        'Class': np.ones(num_fraud, dtype=int)
    }
    for i in range(1, 29):
        fraud_data[f'V{i}'] = np.random.normal(loc=0, scale=1, size=num_fraud)
    
    # Combine non-fraud and fraud data into a DataFrame
    non_fraud_df = pd.DataFrame(non_fraud_data)
    fraud_df = pd.DataFrame(fraud_data)
    data_df = pd.concat([non_fraud_df, fraud_df], ignore_index=True)
    
    # Shuffle the dataset
    data_df = data_df.sample(frac=1).reset_index(drop=True)
    
    # Save to CSV
    data_df.to_csv('mock_data.csv', index=False)
    print("Mock data generated and saved to 'mock_data.csv'")

# Generate mock data
generate_mock_data()
