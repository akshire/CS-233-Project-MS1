import numpy as np

from ..utils import get_n_classes, label_to_onehot, onehot_to_label, accuracy_fn


class LogisticRegression(object):
    """
    Logistic regression classifier.
    """

    def __init__(self, lr = 2.02, max_iters=1228, task_kind = 'classification'):
        """
        Initialize the new object (see dummy_methods.py)
-        and set its arguments.
-
-        Arguments:
-            lr (float): learning rate of the gradient descent
-            max_iters (int): maximum number of iterations
        """
        self.lr = lr
        self.max_iters = max_iters
        self.weights = None
        self.task_kind = task_kind


    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): regression target of shape (N,)
        Returns:
            pred_labels (array): target of shape (N,)
        """
        # Get the number of classes
        n_classes = get_n_classes(training_labels)
        # Convert labels to one-hot encoding
        training_labels_onehot = label_to_onehot(training_labels, n_classes)
        # Get the number of samples and features
        N, D = training_data.shape
        # Initialize the weights
        self.weights = np.zeros((D, n_classes))
        # Gradient descent
        for _ in range(self.max_iters):
            # Compute the logits
            logits = np.dot(training_data, self.weights)
            # Compute the probabilities
            probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
            # Compute the gradients
            grad = np.dot(training_data.T, (probs - training_labels_onehot)) / N
            # Update the weights
            self.weights -= self.lr * grad
        # Predict the labels
        pred_labels = onehot_to_label(probs)
        return pred_labels
        


    

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        # Compute the logits
        logits = np.dot(test_data, self.weights)
        # Compute the probabilities
        probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
        # Predict the labels
        pred_labels = onehot_to_label(probs)
        return pred_labels
        
