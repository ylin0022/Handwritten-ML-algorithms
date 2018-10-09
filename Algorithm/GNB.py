import numpy as np

class GNB:
    
    def __init__(self,C):
        #initialize GNB Classifier
        assert C > 0, 'C must be valid'
        self._C = C # number of class
    
    def fit(self, x, y):
        self._PC = np.zeros(self._C);
        self._mean = np.zeros((self._C, x.shape[1]))
        self._var = np.zeros((self._C, x.shape[1]))
        for i in range(self._C):
            tmp=x[y[0:y.shape[0]] == i]
            self._mean[i] = np.mean( tmp, axis=0 )
            self._var[i] = np.var( tmp, axis=0 )
            self._PC[i] = tmp.shape[0]/x.shape[0] #prior possibility of each class
    
    def cal_GNB(self,x_q):
        C = np.log(self._PC)
        for k in range(self._C):
            for i in range(x_q.shape[0]):
                #calculate possibility of each class and store
                C[k] +=  (-(x_q[i]-self._mean[k][i])**2 / (2 * self._var[k][i]**2) ) - np.log(np.sqrt(2*np.pi* self._var[k][i]**2))
                
        return np.argmax(C)

    def predict(self,x_new):
        class_out = np.zeros(x_new.shape[0])
        for i in range(x_new.shape[0]):
        #    x_q = datatest[i]
            class_out[i] = self.cal_GNB(x_new[i])# calculate possibility of each class
        return class_out