import h5py
import numpy as np
with h5py.File('../Input/images_training.h5','r') as H:
    data = np.copy(H['data'])
    X_train = data.reshape(len(data), -1)
with h5py.File('../Input/labels_training.h5','r') as H:
    label = np.copy(H['label'])
    y_train =label.reshape(len(label), -1)
with h5py.File('../Input/images_testing.h5','r') as H:
    real_test = np.copy(H['data'])
    X_test = real_test.reshape(len(real_test), -1)[:2000,:]
with h5py.File('../Input/labels_testing_2000.h5','r') as H:
    real_label = np.copy(H['label'])
    y_test =real_label.reshape(len(real_label), -1)
import matplotlib.pyplot as plt
import numpy as np
#import seaborn
from mpl_toolkits.mplot3d import Axes3D

X = X_train
new_X = X_train_pca
y = y_train.reshape(30000,)
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(new_X[:, 0], new_X[:, 1], new_X[:, 2], c=y)
plt.show()
