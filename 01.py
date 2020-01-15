import pandas as pd
import numpy as np
df = pd.read_csv(r'Data\test.csv',dtype={'FIELD_45': str})
print(df.head())

print(df.columns[51:53])

df.to_excel(r'Data\test.xlsx', index = False)