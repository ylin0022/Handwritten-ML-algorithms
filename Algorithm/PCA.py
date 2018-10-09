from scipy import linalg

class PCA:
    
    def __init__(self, n_components):
        #初始化PCA
        assert n_components >= 1, 'n_components must be valid'
        self.n_components = n_components
        self.U = None
    
    def fit(self, X ):
        
        assert self.n_components <= X.shape[1], \
        'n_components must not be more than the feature number of X'
        self.U, self.s, self.VT = linalg.svd(X)
        
        return self
        
    def transform(self, X):
        #将给定的X，映射到各个主成分分量上
        assert self.VT.T[:,:self.n_components].shape[0] == X.shape[1], \
        'cannot transform because of dimension error, check'
        assert self.VT.T is not None, 'you must fit before transform'
        
        #return self.VT.T[:,:self.n_components].dot(X)
        return X.dot(self.VT.T[:,:self.n_components])

    def inverse_transform(self, X):
        #将给定的X，反向映射回原来的特征空间
        assert self.VT[:,:self.n_components].shape[0] == X.shape[1]
        
        #return self.VT[:,:self.n_components].dot(X)
        return X.dot(VT.T[:,:self.n_components].T)
    def __repr__(self):
        return 'PCA(n_components = {})'.format(self.n_components)