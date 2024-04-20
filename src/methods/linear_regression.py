import numpy as np
import sys

class LinearRegression(object):
    """
        Linear regressor object. 
        Note: This class will implement BOTH linear regression and ridge regression.
        Recall that linear regression is just ridge regression with lambda=0.
    """

    def __init__(self, lmda, task_kind = "regression"):
        """
            Initialize the task_kind (see dummy_methods.py)
            and call set_arguments function of this class.
        """
        self.lmda = lmda
        self.task_kind = task_kind
        

    
    
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
            toInvert = np.transpose(data) @ data - self.lmda * np.eye(data.shape[1])
            inverted = np.linalg.inv(toInvert)
            W = inverted @ np.transpose(data) @ labels
            return W
        
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
