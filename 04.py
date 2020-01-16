## Do the same thing for test file
import pandas as pd
import numpy as np
import math
import variance_plot as vp

#NOTE: TEST DATA DOES NOT HAVE LABEL COLUMNS, HENCE IT IS NOT INCLUDED IN THE DROPPED COLUMN BEFORE PCA
# REMEMBER TO DROP LABEL COLUMNS WHEN DOING DATA CLEANING PIPELINE FOR TRAIN DATA
_df = pd.read_excel(r'Data\test.xlsx', dtype={'FIELD_45': str})

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

    #Generate Vectorizer for FIELD_7
    word_vector = generate_vectorizer(df,'FIELD_7', 100)
    df.drop(columns = 'FIELD_7', inplace = True)
    df = pd.concat([df, word_vector], axis = 1)


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


    # GET PCA DATA
    import variance_plot as vp
    label = df[['id']]
    df.drop(columns=['maCv', 'province', 'district', 'id'], inplace=True)
    df = pd.get_dummies(df)
    df_std = df.copy()
    from sklearn.preprocessing import StandardScaler
    __ = StandardScaler().fit_transform(df_std)
    df_std = pd.DataFrame(__, columns=df.columns)
    df_pca, df_var = vp.pca_transform(df_std, 100)
    df_pca = pd.concat([label, df_pca], axis=1)

    return df_pca

test = pipeline_data(_df)
test.to_excel(r'Data\Test - Final Data - PCA - 100.xlsx', index = False)