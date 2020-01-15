import pandas as pd
import numpy as np
import math

df = pd.read_excel(r'Data\train.xlsx', dtype={'FIELD_45': str})
df.dropna(axis=1, how='all', inplace = True)
df.info()
print('='*20)
for i, item in enumerate(df):
    try:
        #df['FIELD_11'] is a string series but only have 'number' as values
        # cannot convert the series into num, so have to deal with this column separately
        column = df.iloc[:,i]
        if(~column.equals(df['FIELD_11'])):
            if (np.issubdtype(column.dtype, np.number)):
                for j, item_ in enumerate(column):
                    if(math.isnan(item_)):
                        df.iloc[:,i].iloc[j] = -1
                #print(df.iloc[:,i].dtype.name)
            else:
                for j, item_ in enumerate(column):
                    if(pd.isnull(item_) or item_ == 'None' or item_ is None or item_ == 'na'):
                        df.iloc[:,i].iloc[j] = 'Fill'
        else:
            for j, item_ in enumerate(df['FIELD_11']):
                if (df['FIELD_11'].iloc[j] == '' or math.isnan(df['FIELD_11'].iloc[j])):
                    df['FIELD_11'].iloc[j] = np.nan
                elif (~(df['FIELD_11'].iloc[j] == np.nan)):
                    df['FIELD_11'].iloc[j] = int(df['FIELD_11'].iloc[j])
    except Exception as e:
        print(e)
        # print(item)
    #print('-' * 20)
df['FIELD_11'] = df['FIELD_11'].apply(lambda x: -1 if x == 'Fill' else x)
df.info()
print(df['FIELD_11'].value_counts())
print(df['FIELD_11'].isna().sum())
df['FIELD_11'] = df['FIELD_11'].astype(int)
print(df['FIELD_11'].isna().sum())
df['FIELD_45'] = df['FIELD_45'].fillna(-1)
df['FIELD_45'] = df['FIELD_45'].apply(lambda x: -1 if x == 'Fill' else x)
df['FIELD_45'] = df['FIELD_45'].astype(int)
df.info()
df.to_excel(r'Data\Cleaned NaN.xlsx', index = False)