from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KernelDensity

class KDEClassifer(BaseEstimator, ClassifierMixin):
    def __init__(self, bandwidth = 1 , kernel = 'gaussian', alpha = 1):
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.alpha = alpha
    
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        Xgroup = [X[y == yi] for yi in self.classes_]
        self.log_priors_ = np.log([X[y == yi].shape[0]/X.shape[0] for yi in self.classes_])
        self.kernels_ = [KernelDensity(bandwidth= self.bandwidth, kernel = self.kernel).fit(Xi) for Xi in Xgroup]
        
        return self

    def predict_proba(self, X):
        log_prob = np.array([kernel.score_samples(X) for kernel in self.kernels_]).T
        result = np.exp(log_prob + self.log_priors_)

        return result/ self.alpha
    
    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X),axis = 1)]