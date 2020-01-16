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
df = pd.read_excel(r'Data\Cleaned NaN.xlsx')
word_vector = generate_vectorizer(df,'FIELD_7', 100)
df.drop(columns = 'FIELD_7', inplace = True)
df = pd.concat([df, word_vector], axis = 1)
print(df.shape)


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

df.to_excel(r'Data\Final Data.xlsx', index = False)