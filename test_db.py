import sqlite3
import pandas as pd

DATABASE = 'dash_metrics.db'

# 读取数据
conn = sqlite3.connect(DATABASE)
df = pd.read_sql_query("SELECT * FROM dash_metrics ORDER BY timestamp DESC LIMIT 500", conn)
conn.close()

print(df.head())
