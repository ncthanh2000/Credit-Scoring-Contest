import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

def bar_plot_distribution(_df, _x_feature, _threshold):
	#Vẽ distribution của _x_feature, thuộc _df, với điều kiện count của các row phải 
	# lớn hơn _threshold.
	# Return df_count là 1 dataframe ghi value count của 1 row, value của row đó, với
	# điều kiện value count >= _threshold
    names = _df[_x_feature].value_counts().keys().tolist()
    counts = _df[_x_feature].value_counts().tolist()
    df_count = pd.DataFrame()
    df_count[str(_x_feature)] = names
    df_count['Count'] = counts
    df_count = df_count[df_count['Count'] >= _threshold]

    #Re-order and re-index df_plot, which is a copy of df_count
    df_plot = df_count.sort_values(by ='Count', ascending = False).reindex()
    fig, axes = plt.subplots(ncols = 1, figsize = (20,7))
    sns.barplot(y = df_plot['Count'], x = df_plot[str(_x_feature)],ax = axes)
    	#,order = df_count['Count']
    axes.set_xlabel("Count")
    axes.set_xlabel(str(_x_feature))
    if(len(df_count) > 8):
        axes.set_xticklabels(axes.xaxis.get_majorticklabels(), rotation=90)
    fig.suptitle('Count Distribution of '+str(_x_feature), fontsize=20)
    plt.show()

    return df_count

def proximity(_dataframe, _feature, _threshold):
    names = _dataframe[_feature].value_counts().keys().tolist()
    counts = _dataframe[_feature].value_counts().tolist()
    df_count = pd.DataFrame()
    df_count[_feature] = names
    df_count['Count'] = counts
    total_count = df_count['Count'].sum()
    '''
    df_count['Percentage'] = df_count.apply(lambda row: int(round(100*row['Count']/total_count, 0)), 
                                            axis=1)
    df_count = df_count[df_count['Percentage'] >= _threshhold]
    '''
    df_count = df_count[df_count['Count'] >= _threshold]
    return df_count


def prox(_df, _feature, threshold):
    names = _df[_feature].value_counts().keys().tolist()
    counts = _df[_feature].value_counts().tolist()
    df_count = pd.DataFrame()
    df_count[_feature] = names
    df_count['Count'] = counts
    df_count = df_count[df_count['Count'] >= threshold]
    return df_count