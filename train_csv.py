import pandas as pd
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine

df = pd.read_csv("dataset/HR Data.txt", sep='\t')
print(df.shape)
print(df.columns.tolist())



# engine = create_engine("sqlite:///employee.db")

# db = SQLDatabase(engine=engine)
# df.to_sql("employee", engine, index=False)