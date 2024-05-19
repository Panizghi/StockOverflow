import neurelo_sdk as nrl
import pandas as pd
import random
import string
import psycopg2

# Initialize the Neurelo SDK
nrl.initialize(api_key='YOUR_API_KEY')

# Function to fetch public keys from PostgreSQL
def fetch_public_keys():
    conn = psycopg2.connect("dbname=near_indexer user=user password=password host=localhost")
    cur = conn.cursor()
    cur.execute("SELECT public_key FROM accounts")
    public_keys = cur.fetchall()
    cur.close()
    conn.close()
    return [pk[0] for pk in public_keys]

# Function to generate random public keys if needed
def generate_public_key():
    return ''.join(random.choices(string.ascii_letters + string.digits, k=64))

# Define the schema for mock data
schema = {
    "Time": "timestamp",
    "V1": "float",
    "V2": "float",
    "V3": "float",
    "V4": "float",
    "V5": "float",
    "V6": "float",
    "V7": "float",
    "V8": "float",
    "V9": "float",
    "V10": "float",
    "V11": "float",
    "V12": "float",
    "V13": "float",
    "V14": "float",
    "V15": "float",
    "V16": "float",
    "V17": "float",
    "V18": "float",
    "V19": "float",
    "V20": "float",
    "V21": "float",
    "V22": "float",
    "V23": "float",
    "V24": "float",
    "V25": "float",
    "V26": "float",
    "V27": "float",
    "V28": "float",
    "Amount": "float",
    "PublicKey": "string"
}

# Generate mock data
mock_data = nrl.generate_mock_data(schema, num_records=1000)

# Fetch public keys from the NEAR Indexer for Wallet
public_keys = fetch_public_keys()

# Assign a public key to each mock data record
for record in mock_data:
    record['PublicKey'] = random.choice(public_keys)

# Convert to DataFrame
df = pd.DataFrame(mock_data)

# Save to CSV
df.to_csv('mock_data.csv', index=False)
print("Mock data generated and saved to 'mock_data.csv'")
