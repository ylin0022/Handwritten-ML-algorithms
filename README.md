# README
10-categories classification Machine Learning algorithms: KNN, GNB, LR
-------------
#### Programming Language  
Python 3.6.5  
Jupyter Notebook  
Pytorch(only for CNN_main.ipynb)  

#### Instructions
* The data used is part of fashion-mnist containing a training set of 30000 examples and a test set of 5000 examples.
* The submission contains all the 5000 labels, following the assignment instruction.   
* KNN is the one with highest accuracy, and the output file is predicted by KNN   
* CNN is for reference with existing libraries(of courese) with Pytorch  
1.Find n_components for PCA.ipynb  
2.KNN_main.ipynb  
3.GNB_main.ipynb  
4.LR_main.ipynb  
5.CNN_main.ipynb  

**The algorithms** in each notebook file are packaged in different .py files, so all you need to do is to follow the codes in notebook and run them.  
It can automatically read data from the Input file and calculate the accuracy.
##### Tips to know:
* These are all hand-written codes  
* The following code will create the data after PCA in 'Input' file to make it easier for other training process.
```
with h5py.File('../Input/data_PCA.h5','r') as H:
    data_PCA = np.copy(H['data_PCA'])
with h5py.File('../Input/datatest_PCA.h5','r') as H:
    datatest_PCA = np.copy(H['datatest_PCA'])
```
* CNN_main file needs Pytorch to successfully run, but you can see the training process and results of loss and accuracy.  

-------------
**Several results**
* KNN: 85.25%
* GNB: 44.55%
* LR: 83.40%
* SVM: 0.20%

-------------
--------Created by Yucong and Yang on 2018/10/01
