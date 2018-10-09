import numpy as np

def accuracy(y_predict, y_test):
    #calculate accuracy of y_predict
    assert len(y_test) == len(y_predict), \
        "the size of y_true must be equal to the size of y_predict"
    '''count = 0
    for i in range(len(y_test)):
        if y_predict[i] == y_test[i]:
            count += 1
    return count / len(y_test)'''
    return np.sum(y_test == y_predict) / len(y_test)