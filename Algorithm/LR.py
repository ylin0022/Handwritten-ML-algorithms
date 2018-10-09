import numpy as np

class LR:
    
    def __init__(self,C,regulation,eta,n_iter):
        # initialize LR Classifier
        assert C > 0, 'C must be valid'
        assert eta > 0, 'eta must be valid'
        assert n_iter > 0, 'n_iter must be valid'
        self._C = C
        self._regulation = regulation
        self._eta = eta
        self._n_iter = n_iter
    
    def LR_dloss(self,x,w,y):
        # calculate and return dloss and loss
        z = np.dot(x,w) #N*
        exp_z = np.exp(z)
        p = -exp_z/(1 + exp_z) + y
        dloss = - np.sum(p*x , axis=0)[:, np.newaxis] + self._regulation*w
        loss = -np.sum(-np.log(1 + exp_z) + y*z)
        
        return dloss, loss
    
    def LR_grad(self,x,y):
        N, D = x.shape
        weight = np.zeros(D)[:, np.newaxis]
        for epoch in range(self._n_iter):
            dloss, loss = self.LR_dloss(x,weight, y)
            weight = weight  - dloss * self._eta # gradient descent
        return weight
    
    def fit(self, data, label):
        # train LR classifier with training data and label
        x = np.hstack((np.ones((data.shape[0],1)), data))
        y = np.zeros((label.shape[0],1))
        self._weights = np.zeros((x.shape[1],10))
        
        for i in range(self._C): # calculate for each class
            y[label[0:label.shape[0]] ==i] = 1
            y[label[0:label.shape[0]] !=i] = 0
            self._weights.T[i]=self.LR_grad(x,y).T
    
    def predict(self,datatest):
        x_test = np.hstack((np.ones((datatest.shape[0],1)), datatest))
        class_out = np.zeros(x_test.shape[0])
        for i in range(x_test.shape[0]):
            z = np.dot(x_test[i],self._weights) #N*1
            exp_z = np.exp(z)
            p = exp_z/(1 + exp_z) # sigmoid function
            class_out[i] = np.argmax(p)
        return class_out