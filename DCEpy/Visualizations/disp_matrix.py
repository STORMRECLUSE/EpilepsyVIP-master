
import matplotlib.pyplot as plt
import numpy as np

def disp_mat(data, title, column_labels, row_labels):

	fig, ax = plt.subplots()
	heatmap = ax.pcolor(data, cmap=plt.cm.Blues)

	# put the major ticks at the middle of each cell
	ax.set_xticks(np.arange(data.shape[0])+0.5, minor=False)
	ax.set_yticks(np.arange(data.shape[1])+0.5, minor=False)

	# want a more natural, table-like display
	ax.invert_yaxis()
	ax.xaxis.tick_top()
	ax.set_xticklabels(row_labels, minor=False)
	ax.set_yticklabels(column_labels, minor=False)

	# add numbers 
	for y in range(data.shape[0]):
	    for x in range(data.shape[1]):
	        plt.text(x + 0.5, y + 0.5, '%.4f' % data[y, x],
	                 horizontalalignment='center',
	                 verticalalignment='center',
	                 )

	# plt.colorbar(heatmap)
	plt.title(title, y=1.05)
	plt.show()