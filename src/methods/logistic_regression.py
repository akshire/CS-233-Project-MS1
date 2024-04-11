import numpy as np

from ..utils import get_n_classes, label_to_onehot, onehot_to_label, accuracy_fn


class LogisticRegression(object):
    """
    Logistic regression classifier.
    """

    def __init__(self, lr, max_iters=500):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            lr (float): learning rate of the gradient descent
            max_iters (int): maximum number of iterations
        """
        self.lr = lr
        self.max_iters = max_iters

    
    def f_softmax(self, data, W):
        """
        Softmax function for multi-class logistic regression.
        
        Args:
            data (array): Input data of shape (N, D)
            W (array): Weights of shape (D, C) where C is the number of classes
        Returns:
            array of shape (N, C): Probability array where each value is in the
                range [0, 1] and each row sums to 1.
                The row i corresponds to the prediction of the ith data sample, and 
                the column j to the jth class. So element [i, j] is P(y_i=k | x_i, W)
        """
        
        z = np.dot(data, W)  # Shape: (N, C)
        
        # Apply the softmax function to convert linear scores into probabilities
        exp_scores = np.exp(z)  # Shape: (N, C)
        probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  # Shape: (N, C)
        
        return probabilities

    
    def loss_logistic_multi(self,data, labels, w):
        """ 
        Loss function for multi class logistic regression, i.e., multi-class entropy.
        
        Args:
            data (array): Input data of shape (N, D)
            labels (array): Labels of shape  (N, C)  (in one-hot representation)
            w (array): Weights of shape (D, C)
        Returns:
            float: Loss value 
        """
        
        # Calculate the softmax probabilities
        probs = self.f_softmax(data, w)
        # Compute the cross-entropy loss
        cross_entropy_loss = -np.sum(labels * np.log(probs))
        # Normalize the loss by the number of samples
        loss = cross_entropy_loss / data.shape[0]
        return loss

    
    def gradient_logistic_multi(self,data, labels, W):
        """
        Compute the gradient of the entropy for multi-class logistic regression.
        
        Args:
            data (array): Input data of shape (N, D)
            labels (array): Labels of shape  (N, C)  (in one-hot representation)
            W (array): Weights of shape (D, C)
        Returns:
            grad (np.array): Gradients of shape (D, C)
        """
        print("SOFTMAAAAAAAAAAAAAAAAAAAAAAX: ",self.f_softmax(data,W).shape)
        print("LABEEEEEEEEEEEEEEEEEEEEEEEEEEELS: ", labels.shape)
        print("DATAAAAAAAAAAAAAAAAAAAAAAAA.TTTTTTTTT: ",data.T.shape)
        var = data.T@(self.f_softmax(data,W)-labels)
        print("VAAAAAAAAAAAAAAAAAAAAAAAAAAR: ",var.shape)
        return var

    
    def logistic_regression_predict_multi(self, data, W):
        """
        Prediction the label of data for multi-class logistic regression.
        
        Args:
            data (array): Dataset of shape (N, D).
            W (array): Weights of multi-class logistic regression model of shape (D, C)
        Returns:
            array of shape (N,): Label predictions of data.
        """
        
        # Compute the logits for each class
        logits = np.dot(data, W)
        # Apply softmax to logits to get probabilities
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        # Choose the class with the highest probability as the prediction
        predictions = np.argmax(probabilities, axis=1)
        return predictions



    def logistic_regression_train_multi(self,data, labels):
        """
        Training function for multi class logistic regression.
        
        Args:
            data (array): Dataset of shape (N, D).
            labels (array): Labels of shape (N, C)
        Returns:
            weights (array): weights of the logistic regression model, of shape(D, C)
        """
        print(data.shape)
        D = data.shape[1] # number of features
        print(labels.shape)
        C = 1  # number of classes
        # Random initialization of the weights
        weights = np.random.normal(0, 0.1, (D, C))
        for it in range(self.max_iters):
            # Compute predictions and loss
            print(type(data))
            print(type(weights))
            predictions = self.logistic_regression_predict_multi(data, weights)
            exp_logits = np.exp(np.dot(data, weights) - np.max(np.dot(data, weights), axis=1, keepdims=True))
            y_pred = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            loss = -np.sum(np.log(y_pred)*labels)
            
            # Compute the gradient of the loss with respect to weights
            gradient = self.gradient_logistic_multi(data, labels, weights)
            print("WEIGHTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTS: ",weights.shape)
            print("GRADIEEEEEEEEEEEEEEEEEEEEEEEEEENT : ", gradient.shape)
            # Update the weights
            weights -= self.lr * gradient
    
    
            predictions = self.logistic_regression_predict_multi(data, weights)
            if accuracy_fn(predictions, onehot_to_label(labels)) == 100:
                break
            
        return weights

    
    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): regression target of shape (N,)
        Returns:
            pred_labels (array): target of shape (N,)
        """
        ##
        ###
        #### WRITE YOUR CODE HERE!
        ###
        ##
        weights = self.logistic_regression_train_multi(training_data, training_labels)
        training_data@weights
        
        
        return pred_labels


    

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        ##
        ###
        #### WRITE YOUR CODE HERE!
        ###
        ##
        return pred_labels
