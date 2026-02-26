import pandas as pd
from sqlalchemy import create_engine, text
import os

# Koneksi ke database
engine = create_engine('postgresql://analyst:analyst123@localhost/ecommerce_analytics')

# Load CSV files
df_users = pd.read_csv('data/raw/users.csv')
df_products = pd.read_csv('data/raw/products.csv')
df_orders = pd.read_csv('data/raw/order_items.csv')

# Simpan ke database
df_users.to_sql('users', engine, if_exists='replace', index=False)
df_products.to_sql('products', engine, if_exists='replace', index=False)
df_orders.to_sql('order_items', engine, if_exists='replace', index=False)

print("✅ Data loaded to PostgreSQL!")

# Test query
with engine.connect() as conn:
    result = conn.execute(text("SELECT COUNT(*) FROM users"))
    count = result.scalar()
    print(f"Users in database: {count}")