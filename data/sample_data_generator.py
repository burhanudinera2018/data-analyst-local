import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed
np.random.seed(42)
random.seed(42)

print("Generating sample ecommerce data...")

# ==================== USERS TABLE ====================
n_users = 5000
users = []
for i in range(1, n_users + 1):
    users.append({
        'id': i,
        'first_name': f'User{i}',
        'last_name': f'LastName{i}',
        'email': f'user{i}@example.com',
        'created_at': datetime.now() - timedelta(days=random.randint(30, 1000))
    })
df_users = pd.DataFrame(users)
df_users.to_csv('data/raw/users.csv', index=False)
print(f"Created {n_users} users")

# ==================== PRODUCTS TABLE ====================
n_products = 500
categories = ['Electronics', 'Clothing', 'Home', 'Sports', 'Books', 'Toys']
products = []
for i in range(1, n_products + 1):
    products.append({
        'id': i,
        'name': f'Product {i}',
        'category': random.choice(categories),
        'brand': f'Brand{random.randint(1, 20)}',
        'price': round(np.random.uniform(10, 500), 2),
        'cost': round(np.random.uniform(5, 300), 2)
    })
df_products = pd.DataFrame(products)
df_products.to_csv('data/raw/products.csv', index=False)
print(f"Created {n_products} products")

# ==================== ORDER ITEMS TABLE ====================
n_orders = 50000
order_items = []
statuses = ['Complete', 'Pending', 'Cancelled', 'Refunded']

for i in range(1, n_orders + 1):
    user_id = random.randint(1, n_users)
    product_id = random.randint(1, n_products)
    quantity = random.randint(1, 5)
    product_price = df_products[df_products['id'] == product_id]['price'].values[0]
    
    order_items.append({
        'id': i,
        'order_id': f'ORD{i:06d}',
        'user_id': user_id,
        'product_id': product_id,
        'quantity': quantity,
        'sale_price': round(product_price * quantity, 2),
        'status': random.choices(statuses, weights=[0.8, 0.1, 0.05, 0.05])[0],
        'created_at': datetime.now() - timedelta(days=random.randint(1, 730))
    })

df_orders = pd.DataFrame(order_items)
df_orders.to_csv('data/raw/order_items.csv', index=False)
print(f"Created {n_orders} order items")

print("\n✅ Data generation complete!")
print(f"Files saved in data/raw/:")
print(f"- users.csv: {len(df_users)} records")
print(f"- products.csv: {len(df_products)} records")
print(f"- order_items.csv: {len(df_orders)} records")