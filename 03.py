import pandas as pd
import numpy as np
import math
df = pd.read_excel(r'Data\Cleaned NaN.xlsx')
print(df['FIELD_11'].value_counts())
print(df['FIELD_11'].isna().sum())
column = df['FIELD_11']
for i in range(len(column)):
    if (column.iloc[i] == '' or math.isnan(df['FIELD_11'].iloc[i])):
        df['FIELD_11'].iloc[i] =-1
    else:
        try:
            df['FIELD_11'].iloc[i] = int(df['FIELD_11'].iloc[i])
        except:
            print(df['FIELD_11'].iloc[i] == np.nan)
df['FIELD_11'] = df['FIELD_11'].astype(int)
print(df['FIELD_11'].isna().sum())
df['FIELD_45'] = df['FIELD_45'].fillna(-1)
df['FIELD_45'] = df['FIELD_45'].astype(int)
df.to_excel(r'Data\Cleaned NaN - 01.xlsx', index = False)