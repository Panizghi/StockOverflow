import pandas as pd
import numpy as np
import xgboost as xgb
import psycopg2

def fetch_new_data(conn):
    query = """
    SELECT id, Time, V1, V2, V3, V4, V5, V6, V7, V8, V9, V10,
           V11, V12, V13, V14, V15, V16, V17, V18, V19,
           V20, V21, V22, V23, V24, V25, V26, V27, V28, Amount
    FROM transactions
    WHERE predicted_class IS NULL;
    """
    return pd.read_sql(query, conn)

def update_predictions(conn, ids, predictions):
    query = """
    UPDATE transactions
    SET predicted_class = %s
    WHERE id = %s;
    """
    with conn.cursor() as cur:
        for id, pred in zip(ids, predictions):
            cur.execute(query, (pred, id))
    conn.commit()

def main():
    # Database connection
    conn = psycopg2.connect("dbname=near_indexer user=user password=password host=localhost")

    # Fetch new data
    new_data = fetch_new_data(conn)
    
    if new_data.empty:
        print("No new data to process.")
        return
    
    # Preprocess data
    predictors = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                  'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
                  'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']
    
    dmatrix = xgb.DMatrix(new_data[predictors])
    
    # Load the model
    model = xgb.Booster()
    model.load_model('fraud_models/xgb_model.json')
    
    # Predict
    predictions = model.predict(dmatrix)
    predictions = np.where(predictions > 0.5, 1, 0)  # Classify as fraud if prediction > 0.5
    
    # Update the database with predictions
    update_predictions(conn, new_data['id'].values, predictions)
    print("Predictions updated in the database.")

    # Close the connection
    conn.close()

if __name__ == "__main__":
    main()
