import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN, AgglomerativeClustering


def cluster_label_plot(_df, _model):

	if(_model.__class__.__name__ == 'KMeans'):
		new_df = cluster_labelling(_df, _model)
		cluster_plot(new_df, _model)
		return new_df
	else:
		new_df = cluster_labelling_not_knn(_df, _model)
		cluster_plot_not_knn(new_df, _model)
		return new_df


def cluster_labelling(_df, _model):
	_df_copy = _df.copy()
	_model.fit(_df_copy)
	predict = _model.predict(_df_copy)
	_df_copy['Predicted Label-' + _model.__class__.__name__] = pd.Series(predict, index=_df_copy.index)
	return _df_copy

def cluster_plot(_df, _model):
	fig = plt.figure(figsize = (10,6))
	ax = fig.add_subplot(1,1,1) 
	ax.set_xlabel('Principal Component 1', fontsize = 15)
	ax.set_ylabel('Principal Component 2', fontsize = 15)
	ax.set_title(str(_model.__class__.__name__), fontsize = 20)
	targets = [0,1]
	colors = ['r', 'g']
	for target, color in zip(targets,colors):
	    indicesToKeep = _df['Predicted Label-' + _model.__class__.__name__] == target
	    ax.scatter(_df.loc[indicesToKeep, 'principal component 1']
	               , _df.loc[indicesToKeep, 'principal component 2']
	               , c = color
	               , s = 50)
	ax.scatter(_model.cluster_centers_[:,0] , _model.cluster_centers_[:,1], color = colors,
	           marker = '*',  edgecolor='black', s = 300)
	ax.legend(targets)
	ax.grid()
	plt.show()


def cluster_labelling_not_knn(_df, _model):
	_df_copy = _df.copy()
	predict = _model.fit_predict(_df_copy)
	_df_copy['Predicted Label-' + _model.__class__.__name__] = pd.Series(predict, index=_df_copy.index)
	return _df_copy

def cluster_plot_not_knn(_df, _model):
	fig = plt.figure(figsize = (10,6))
	ax = fig.add_subplot(1,1,1) 
	ax.set_xlabel('Principal Component 1', fontsize = 15)
	ax.set_ylabel('Principal Component 2', fontsize = 15)
	ax.set_title(str(_model.__class__.__name__), fontsize = 20)
	targets = [-1,0,1]
	colors = ['k','r', 'g']
	for target, color in zip(targets,colors):
	    indicesToKeep = _df['Predicted Label-' + _model.__class__.__name__] == target
	    ax.scatter(_df.loc[indicesToKeep, 'principal component 1']
	               , _df.loc[indicesToKeep, 'principal component 2']

	               , c = color
	               , s = 50)
	ax.legend(targets)
	ax.grid()
	plt.show()


def cluster_label_only(_df, _model):
	if(_model.__class__.__name__ == 'KMeans'):
		new_df = cluster_labelling(_df, _model)
		return new_df
	else:
		new_df = cluster_labelling_not_knn(_df, _model)
		return new_df	