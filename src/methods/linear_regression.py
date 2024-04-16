import numpy as np
import sys

class LinearRegression(object):
    """
        Linear regressor object. 
        Note: This class will implement BOTH linear regression and ridge regression.
        Recall that linear regression is just ridge regression with lambda=0.
    """

    def __init__(self, lmda, epochs=15000):
        """
            Initialize the task_kind (see dummy_methods.py)
            and call set_arguments function of this class.
        """
        self.lmda = lmda
        self.epochs = epochs #nrb of tries
        

    
    
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
            w = np.random.normal(0, 1e-1, [data.shape[1],labels.shape[1]])
            for try_Nbr in range(self.epochs):
                Y = data @ w
                print("W shape :",w.shape)
                print("data shape :",data.shape)
                print("Y_hat shape :",Y.shape)
                gradient = 1/data.shape[0] * 2 * np.sum(np.transpose(data) @ (Y-labels),axis=0)
                print("gradient shape :",gradient.shape)
                w = w - self.lmda * gradient

            return w
        
        print("XTRAIN SHAPE : ",training_data.shape)
        print("YTRAIN SHAPE : ",training_labels.shape)        
        self.W = get_W(training_data,training_labels)
        pred_regression_targets = training_data @ self.W
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
        pred_regression_targets = test_data @ self.W
        

        return pred_regression_targets
