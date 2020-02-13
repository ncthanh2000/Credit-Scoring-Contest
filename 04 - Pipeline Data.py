## Fill NaN values,  vectorize maCv, Create features based on count of NaN values for each row
import pandas as pd
import numpy as np
import math
import variance_plot as vp

#NOTE: TEST DATA DOES NOT HAVE LABEL COLUMNS, HENCE IT IS NOT INCLUDED IN THE DROPPED COLUMN BEFORE PCA
# REMEMBER TO DROP LABEL COLUMNS WHEN DOING DATA CLEANING PIPELINE FOR TRAIN DATA
_df = pd.read_excel(r'Data\train.xlsx', dtype={'FIELD_45': str})
__df = pd.read_excel(r'Data\test.xlsx', dtype={'FIELD_45': str})
def pipeline_data(df):
    import pandas as pd
    import numpy as np
    import math
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
                elif (column.dtype == bool):
                    df.iloc[:, i] = column.map(lambda x: 1 if x == True else 0)
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

    col = ['FIELD_29', 'FIELD_30', 'FIELD_31', 'FIELD_36', 'FIELD_37']
    for i, item in enumerate(df):
        try:
            # df['FIELD_11'] is a string series but only have 'number' as values
            # cannot convert the series into num, so have to deal with this column separately
            if (df.columns[i] in col):
                print(df.columns[i])
                column = df.iloc[:, i]
                df.iloc[:, i] = column.map(lambda x: 1 if x == 'TRUE' else 0 if x == 'FALSE' else -1)
                df.iloc[:, i] = df.iloc[:, i].astype(int)

        except Exception as e:
            print(e)
            pass

    import pandas as pd
    import numpy as np
    from sklearn.feature_extraction.text import CountVectorizer

    def generate_vectorizer(_df, _col, _max_features):
        import pandas as pd
        import numpy as np
        from sklearn.feature_extraction.text import CountVectorizer
        cv = CountVectorizer(max_features= _max_features, analyzer='word', lowercase=False)

        df_mcv = _df[[_col]]
        #Remove Enclosing Brackets ("[]")
        df_mcv[_col] = df_mcv[_col].apply(lambda x: x[1:-1])

        #Convert to list, with 1 string (x is a string)
        df_mcv[_col] = df_mcv[_col].apply(lambda x: [x])

        df_mcv[_col] = df_mcv[_col].apply(lambda x: " ".join(x))

        X_train = cv.fit_transform(df_mcv[_col])
        X_train = pd.DataFrame(X_train.toarray(), columns=cv.get_feature_names())
        col_key = X_train.columns
        col_val = [_col +' - ' + s for s in col_key]

        dict_ = dict(zip(col_key, col_val))
        print(dict)
        X_train.rename(columns=dict_, inplace=True)
        return X_train

    def CountEncoding(df, _col):
        _df = df[[_col]]
        import pandas as pd
        import category_encoders as ce
        # use binary encoding to encode two categorical features
        enc = ce.CountEncoder(cols=[_col]).fit(_df)
        # transform the dataset
        numeric_dataset = enc.transform(_df)
        return numeric_dataset



    num = [-1, 0, 1, 2, 3, 4, 5] #Map to
    arab = ['Fill', 'Zero', 'One', 'Two', 'Three', 'Four', 'Five'] #FIELD_35, 44
    latin = ['Fill', 'Fill', 'I', 'II', 'III', 'IV', 'V'] #41

    crap = ['Fill', 'Zezo', 'One'] #42

    i = 0
    df['FIELD_42'] = df['FIELD_42'].apply(lambda x: num[i] if x == crap[i] else num[i+1] if x == crap[i+1] else num[i+2])
    df['FIELD_35'] = df['FIELD_35'].apply(lambda x: num[i] if x == arab[i] else num[i+1] if x == arab[i+1] else num[i+2] if x == arab[i+2]else num[i+3] if x == arab[i+3]else num[i+4] if x == arab[i+4] else num[i+5] if x == arab[i+5]else num[i+6])
    df['FIELD_41'] = df['FIELD_41'].apply(lambda x: num[i] if x == latin[i] else num[i+1] if x == latin[i+1] else num[i+2] if x == latin[i+2]else num[i+3] if x == latin[i+3]else num[i+4] if x == latin[i+4] else num[i+5] if x == latin[i+5]else num[i+6])
    df['FIELD_44'] = df['FIELD_44'].apply(lambda x: num[i] if x == arab[i] else num[i] if x == arab[i+1] else num[i+2] if x == arab[i+2]else num[i+3] if x == arab[i+3]else num[i+4] if x == arab[i+4] else num[i+5] if x == arab[i+5]else num[i+6])

    df['FIELD_35'] = df['FIELD_35'].astype(int)
    df['FIELD_41'] = df['FIELD_41'].astype(int)
    df['FIELD_42'] = df['FIELD_42'].astype(int)
    df['FIELD_44'] = df['FIELD_44'].astype(int)

    #Calculate total number of missing values
    df_num = df.select_dtypes(include=[np.number])
    df_text = df.select_dtypes(exclude=[np.number])

    df_num['Num NaN Count'] = 0
    for i, item in df_num.iterrows():
        df_num['Num NaN Count'].iloc[i] = np.sum((df_num.iloc[i] == -1).values.ravel())

    df_text['String NaN Count'] = 0
    for i, item in df_text.iterrows():
        df_text['String NaN Count'].iloc[i] = np.sum(df_text.iloc[i].str.count('Fill'))

    df = pd.concat([df_num, df_text], axis = 1)
    df['Total NaN'] = df.apply(lambda x: x['String NaN Count'] + x['Num NaN Count'],  axis = 1)
    df_fillNa = df.copy()

    # Generate Vectorizer for FIELD_7
    word_vector = generate_vectorizer(df, 'FIELD_7', 100)
    df.drop(columns='FIELD_7', inplace=True)
    df = pd.concat([df, word_vector], axis=1)

    return df

df_test_fillNa = pipeline_data(__df)
df_train_fillNa = pipeline_data(_df)

# df_val_fillNa = df_train_fillNa.iloc[29000:]
# df_train_fillNa = df_train_fillNa.iloc[:29000]

# val_label = df_val_fillNa[['label']]
# val_id = df_val_fillNa[['id']]

df_train_fillNa.drop(columns=['maCv', 'province', 'district'], inplace=True)
df_test_fillNa.drop(columns=['maCv', 'province', 'district'], inplace=True)

df_test_fillNa = pd.get_dummies(df_test_fillNa)
df_train_fillNa = pd.get_dummies(df_train_fillNa)

columns_in_test = list(set(df_test_fillNa.columns.tolist()).difference(set(df_train_fillNa.columns.tolist())))
columns_in_train = list(set(df_train_fillNa.columns).difference(set(df_test_fillNa.columns)))


print(columns_in_test)
print(columns_in_train)
print("helloooooooooooooooooooooooooooooooooooooooooooooooooooooo")
columns_in_train.remove('label')
print(columns_in_train)
print(list(set(df_test_fillNa.columns.tolist()).symmetric_difference(set(df_train_fillNa.columns.tolist()))))

#Train Data
#Drop Mismatch Columns
#df_train_fillNa.drop(columns = 'label', inplace = True)
df_train_fillNa.drop(columns = columns_in_train, inplace = True)
df_test_fillNa.drop(columns = columns_in_test, inplace = True)
print(list(set(df_test_fillNa.columns.tolist()).symmetric_difference(set(df_train_fillNa.columns.tolist()))))



#Train validation split
df_val_fillNa = df_train_fillNa.iloc[29000:]
df_train_fillNa = df_train_fillNa.iloc[:29000]
train_label = df_train_fillNa[['id','label']]
val_label = df_val_fillNa[['id','label']]
print(val_label.head(3))
df_train_fillNa.drop(columns = ['id','label'], inplace = True)
df_val_fillNa.drop(columns = ['id','label'], inplace = True)
test_id = df_test_fillNa[['id']]
df_test_fillNa.drop(columns = 'id', inplace = True)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(df_train_fillNa)
__ = scaler.transform(df_train_fillNa)
df_train_std = pd.DataFrame(__, columns=df_train_fillNa.columns)
df_train_std = pd.concat([train_label, df_train_std], axis=1)


#Test Data

__ = scaler.transform(df_test_fillNa)
df_test_std = pd.DataFrame(__, columns=df_test_fillNa.columns)
df_test_std = pd.concat([test_id, df_test_std], axis=1)

#Validation Data


__ = scaler.transform(df_val_fillNa)
df_val_std = pd.DataFrame(__, columns=df_val_fillNa.columns)
df_val_std.reset_index(drop = True, inplace = True)
val_label.reset_index(drop = True, inplace = True)
df_val_std = pd.concat([val_label, df_val_std],axis=1)

print(df_val_std['label'].value_counts())


df_train_fillNa.to_excel(r'Final Data/Train - FillNa Data.xlsx', index = False)
df_train_std.to_excel(r'Final Data/Train - Std Data.xlsx', index = False)

df_test_fillNa.to_excel(r'Final Data/Test - FillNa Data.xlsx', index = False)
df_test_std.to_excel(r'Final Data/Test - Std Data.xlsx', index = False)


df_val_fillNa.to_excel(r'Final Data/Validate - FillNa Data.xlsx', index = False)
df_val_std.to_excel(r'Final Data/Validate - Std Data.xlsx', index = False)

print(len(val_label))