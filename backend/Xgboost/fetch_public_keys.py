import psycopg2

def fetch_public_keys():
    conn = psycopg2.connect("dbname=near_indexer user=user password=password host=localhost")
    cur = conn.cursor()
    cur.execute("SELECT public_key FROM accounts")
    public_keys = cur.fetchall()
    cur.close()
    conn.close()
    return public_keys

public_keys = fetch_public_keys()
with open('public_keys.txt', 'w') as f:
    for key in public_keys:
        f.write(f"{key[0]}\n")
print("Public keys fetched and saved to 'public_keys.txt'")
