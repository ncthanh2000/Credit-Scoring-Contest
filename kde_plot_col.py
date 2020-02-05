import seaborn as sns
import matplotlib.pyplot as plt
def kde_col(_df, _col, _size):
	row_count = len(_col) + 1
	box, axes = plt.subplots(nrows = row_count, ncols = 2, figsize=(15,_size*row_count))
	box.patch.set_facecolor('white')
	for i in range(len(_col)):
		box = sns.kdeplot(data = _df[_col[i]], ax = axes[int(i/2)][int(i%2)])
		box.set_xlabel(str(_col[i]))

	plt.show()
