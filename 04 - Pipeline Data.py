## Fill NaN values,  vectorize maCv, Create features based on count of NaN values for each row
import pandas as pd
import numpy as np
import math
import variance_plot as vp

#NOTE: TEST DATA DOES NOT HAVE LABEL COLUMNS, HENCE IT IS NOT INCLUDED IN THE DROPPED COLUMN BEFORE PCA
# REMEMBER TO DROP LABEL COLUMNS WHEN DOING DATA CLEANING PIPELINE FOR TRAIN DATA
_df = pd.read_excel(r'Data\train.xlsx', dtype={'FIELD_45': str})
__df = pd.read_excel(r'Data\test.xlsx', dtype={'FIELD_45': str})
def pipeline_data(df, _dropLabel):
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

    # GET PCA DATA
    if(_dropLabel == True):

        columns_in_test = ['FIELD_7 - AT', 'FIELD_7 - HG', 'FIELD_7 - HK', 'FIELD_7 - ND',
       'FIELD_7 - QT', 'FIELD_7 - TL', 'FIELD_7 - XB', 'FIELD_9_74',
       'FIELD_9_CK', 'FIELD_9_TL', 'FIELD_12_DK', 'FIELD_12_DN', 'FIELD_12_DT',
       'FIELD_12_GD', 'FIELD_12_XK', 'FIELD_13_12', 'FIELD_13_A1',
       'FIELD_13_AE', 'FIELD_13_BJ', 'FIELD_13_BW', 'FIELD_13_CG',
       'FIELD_13_CL', 'FIELD_13_DL', 'FIELD_13_DM', 'FIELD_13_EK',
       'FIELD_13_EU', 'FIELD_13_FT', 'FIELD_13_H2', 'FIELD_13_H5',
       'FIELD_13_H7', 'FIELD_13_IS', 'FIELD_13_KX', 'FIELD_13_N7',
       'FIELD_13_NM', 'FIELD_13_NP', 'FIELD_13_NY', 'FIELD_13_QQ',
       'FIELD_13_QS', 'FIELD_13_SB', 'FIELD_13_SZ', 'FIELD_13_ZA',
        'FIELD_39_AN', 'FIELD_39_AO','FIELD_39_AT', 'FIELD_39_CH',
        'FIELD_39_ID', 'FIELD_39_WS']
        for i in range(len(columns_in_test)):
            df[columns_in_test[i]] = 0
        #Train Data
        label = df[['label']]
        df.drop(columns = 'label', inplace = True)
        import variance_plot as vp
        id = df[['id']]
        df.drop(columns=['maCv', 'province', 'district', 'id'], inplace=True)
        df = pd.get_dummies(df)

        df_std = df.copy()
        from sklearn.preprocessing import StandardScaler
        __ = StandardScaler().fit_transform(df_std)
        df_std = pd.DataFrame(__, columns=df.columns)
        df_pca, df_var = vp.pca_transform(df_std, 100)

        df_std = pd.concat([label, df_std], axis=1)
        df_pca = pd.concat([label, df_pca], axis=1)

        df_std = pd.concat([id, df_std], axis=1)
        df_pca = pd.concat([id, df_pca], axis=1)
    else:
        #Test Data
        columns_in_train = ['FIELD_7 - LS', 'FIELD_7 - QN', 'FIELD_9_79', 'FIELD_9_80',
       'FIELD_9_86', 'FIELD_9_MS', 'FIELD_9_NO', 'FIELD_9_XN', 'FIELD_12_TN',
       'FIELD_13_8', 'FIELD_13_AY', 'FIELD_13_BU', 'FIELD_13_CC',
       'FIELD_13_CD', 'FIELD_13_CH', 'FIELD_13_CJ', 'FIELD_13_CR',
       'FIELD_13_CZ', 'FIELD_13_DB', 'FIELD_13_DC', 'FIELD_13_DF',
       'FIELD_13_DI', 'FIELD_13_DQ', 'FIELD_13_DW', 'FIELD_13_E2',
       'FIELD_13_EE', 'FIELD_13_EG', 'FIELD_13_EH', 'FIELD_13_EI',
       'FIELD_13_EL', 'FIELD_13_EN', 'FIELD_13_F1', 'FIELD_13_FN',
       'FIELD_13_FS', 'FIELD_13_FU', 'FIELD_13_FV', 'FIELD_13_GB',
       'FIELD_13_H1', 'FIELD_13_HY', 'FIELD_13_IC', 'FIELD_13_NT',
       'FIELD_13_NW', 'FIELD_13_QU', 'FIELD_13_QX', 'FIELD_13_SE',
       'FIELD_13_SF', 'FIELD_13_SI', 'FIELD_13_SN', 'FIELD_13_SP',
       'FIELD_13_SR', 'FIELD_13_SV', 'FIELD_13_YF', 'FIELD_17_G2',
       'FIELD_39_AD', 'FIELD_39_AE', 'FIELD_39_BE',
       'FIELD_39_CA', 'FIELD_39_GB', 'FIELD_39_IT', 'FIELD_39_N',
       'FIELD_39_SC', 'FIELD_39_SE', 'FIELD_39_TK', 'FIELD_39_TL',
       'FIELD_39_TR', 'FIELD_39_TS', 'FIELD_40_05 08 11 02', 'FIELD_40_08 02',
       'FIELD_40_4']
        for i in range(len(columns_in_train)):
            df[columns_in_train[i]] = 0

        import variance_plot as vp
        id = df[['id']]
        df.drop(columns=['maCv', 'province', 'district', 'id'], inplace=True)
        df = pd.get_dummies(df)
        df_std = df.copy()
        from sklearn.preprocessing import StandardScaler
        __ = StandardScaler().fit_transform(df_std)
        df_std = pd.DataFrame(__, columns=df.columns)
        df_pca, df_var = vp.pca_transform(df_std, 100)

        df_std = pd.concat([id, df_std], axis=1)
        df_pca = pd.concat([id, df_pca], axis=1)
    return df_fillNa,df_std,df_pca

df_train_fillNa,df_train_std,df_train_pca = pipeline_data(_df, True)
df_train_fillNa.to_excel(r'Final Data/Train - FillNa Data.xlsx', index = False)
df_train_pca.to_excel(r'Final Data/Train - PCA - 100.xlsx', index = False)
df_train_std.to_excel(r'Final Data/Train - Std Data.xlsx', index = False)

df_test_fillNa,df_test_std,df_test_pca = pipeline_data(__df, False)
df_test_fillNa.to_excel(r'Final Data/Test - FillNa Data.xlsx', index = False)
df_test_pca.to_excel(r'Final Data/Test - PCA - 100.xlsx', index = False)
df_test_std.to_excel(r'Final Data/Test - Std Data.xlsx', index = False)