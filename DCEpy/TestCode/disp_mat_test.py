import numpy as np 
from DCEpy.Visualizations.disp_matrix import disp_mat


labels = ['Ictal', 'Interictal', 'Postictal', 'Preictal']

# The good
m1 = np.array([[46,0, 2, 0],
				[14, 20,2,12],
				[0,12,36,0],
				[11,1,13,23]])
title = 'AR Coeff and Nonlinear + Radial SVM'
disp_mat(m1, title, labels, labels)

# The bad
m2 = np.array([[12,0,0,36],
				[0,0,0,48],
				[1,0,1,46],
				[0,0,0,48]])
title = 'AR Prediction Error + Poly SVM'
disp_mat(m2, title, labels, labels)

# The ugly
m3_1 = np.array([[42,6,0,0],
				[2,46,0,0],
				[0,1,32,15],
				[1,26,7,14]])
title = 'AR Coeffecients + Radial SVM (1)'
disp_mat(m3_1, title, labels, labels)

m3_2 = np.array([[42,1,4,1],
				[23,17,1,7],
				[0,12,35,1],
				[21,1,12,14]])
title = 'AR Coeffecients + Radial SVM (2)'
disp_mat(m3_2, title, labels, labels)

m3_3 = np.array([[26,18,0,4],
				[14,23,11,0],
				[1,7,27,13],
				[2,21,14,11]])
title = 'AR Coeffecients + Radial SVM (3)'
disp_mat(m3_3, title, labels, labels)
