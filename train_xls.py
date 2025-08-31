import pandas as pd
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine

df = pd.read_csv("dataset/BP4 - Agency Resourcing Table - 2024-25 Budget.csv")
print(df.shape)
print(df.columns.tolist())



engine = create_engine("sqlite:///budget.db")

db = SQLDatabase(engine=engine)
df.to_sql("budget", engine, index=False)