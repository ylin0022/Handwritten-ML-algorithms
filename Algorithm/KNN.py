import numpy as np
from math import sqrt
from collections import Counter
from metrics import accuracy

class KNNClassifier:
    
    def __init__(self, k):
        #initialize KNN Classifier
        assert k >= 1, 'k must be valid'
        self.k = k
        self.X_train = None
        self.y_train = None
    
    def fit(self, X_train, y_train):
        # train KNNClassifier with X_train and y_train 
        assert X_train.shape[0] == y_train.shape[0], \
        'size of X and must be equal to that of y'
        assert self.k <= X_train.shape[0], \
        'size of X_train must be more than k'
        
        self.X_train = X_train
        self.y_train = y_train
        return self
    
    def predict(self, X_predict):
        # given X_predict and return the predicted vectors
        assert self.X_train is not None and self.y_train is not None, \
        'you must fit before predict'
        assert X_predict.shape[1] == self.X_train.shape[1], \
        'the feature number of X_predict must be equal to that of X_train'
        y_predict = [self._predict(x) for x in X_predict]
        return np.array(y_predict)
    
    def _predict(self, x):
        # predict a single x and return its predicted label
        assert x.shape[0] == self.X_train.shape[1], \
        'the feature number of x must be equal to that of X_train'
        
        distances = [sqrt(np.sum((x_train - x)**2)) for x_train in self.X_train]
        near_index = np.argsort(distances)
        topK_y = [int(self.y_train[i]) for i in near_index[:self.k]]
        votes = Counter(topK_y) # calculate number of each labels
        
        return votes.most_common(1)[0][0] # the label with highest votes
    
    def score(self, X_test, y_test):
        
        y_predict = self.predict(X_test)
        return accuracy(y_predict.reshape(-1,1), y_test)
    
    def __repr__(self):
        return 'KNN(k = {})'.format(self.k)
        '''
        classCount = {}
        for i in range(self.k):
            vote = self.y_train[near_index[i]]
            classCount[vote] = classCount.get(vote, 0) + 1
        sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
        return sortedClassCount[0][0]
        '''