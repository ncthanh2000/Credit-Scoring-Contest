import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

def generate_col_names(_pca_df):
    # Takes in a pca-transformed dataframe (which is a numpy array)
    # and generate column names
    names = []
    ncols = _pca_df.shape[1]
    for i in range(1, ncols + 1):
        names.append('principal component ' + str(i))
    return names

def plot_var(_var):
    col = list(range(len(_var)))
    for i in range(len(col)):
        col[i]+=1
    plt.plot( col, _var)
    plt.ylabel("Variance explained Ratio")
    plt.xlabel("Principal Component")
    plt.show()
    return col

def pca_transform(_df, _n):
    
    pca = PCA(n_components=_n)
    df_pca = pca.fit_transform(_df)
    df_pca = pd.DataFrame(data = df_pca
                 , columns = generate_col_names(df_pca))  
    var = pca.explained_variance_ratio_
    
    
    col = plot_var(var)
    df_var = pd.DataFrame(var,col)
    return df_pca, df_var