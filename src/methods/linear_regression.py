import numpy as np
import sys

class LinearRegression(object):
    """
        Linear regressor object. 
        Note: This class will implement BOTH linear regression and ridge regression.
        Recall that linear regression is just ridge regression with lambda=0.
    """

    def __init__(self, lmda, fit, epochs):
        """
            Initialize the task_kind (see dummy_methods.py)
            and call set_arguments function of this class.
        """
        self.lmda = lmda
        self.epochs = epochs #nrb of tries
        self.W = fit
        

    
    
    def fit(self, training_data, training_labels):
        """
            Trains the model, returns predicted labels for training data.
            Arguments:
                training_data (np.array): training data of shape (N,D)
                training_labels (np.array): regression target of shape (N,regression_target_size)
            Returns:
                pred_labels (np.array): target of shape (N,regression_target_size)
        """
        def get_W(data,labels):
            w = np.random.normal(0, 1e-1, data.shape[1])
            for try_Nbr in range(self.epochs):
                y_hat = np.sum(np.transpose(w) * data,axis=1)
                gradient = 1/data.shape[0] * np.sum((y_hat-labels)[:,np.newaxis]*data,axis=0)
                w = w - self.lmda * gradient

            return w
        

        pred_regression_targets = get_W(training_data,training_labels)
        return pred_regression_targets


def predict(self, test_data):
        """
            Runs prediction on the test data.
            
            Arguments:
                test_data (np.array): test data of shape (N,D)
            Returns:
                test_labels (np.array): labels of shape (N,regression_target_size)
        """
        ##
        ###
        #### YOUR CODE HERE!
        ###
        ##
        W = self.W
        pred_regression_targets = np.sum(np.transpose(W) * test_data,axis=1)
        

        return pred_regression_targets
