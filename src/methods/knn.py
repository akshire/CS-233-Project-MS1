import numpy as np

class KNN(object):
    """
        kNN classifier object.
    """

    def __init__(self, k=1, task_kind = "classification"):
        """
            Call set_arguments function of this class.
        """
        self.k = k
        self.task_kind = task_kind

    def euclidean_dist(self, example, training_examples):
        """Compute the Euclidean distance between a single example
        vector and all training_examples.

        Inputs:
            example: shape (D,)
            training_examples: shape (NxD) 
        Outputs:
            euclidean distances: shape (N,)
        """
        
        return np.sqrt(((training_examples - example) ** 2).sum(axis=1))

    def label_find(self, test_point):
        """
            Return the predicted label by calculating all the distances between the point and all the 
            training_data and then returning the most frequent one from k nearest neighbors

            Arguments:
                test_point (np.array): point that we want to label (,D)
            Outputs:
                predicted label: label of the shape
        """
        
        euclid_distances = self.euclidian_dist(self, test_point, self.train_data)
        kNearest = np.argpartition(euclid_distances, self.k)
        labels = np.zeros(kNearest.size)
        index_kNearest = 0
        for j in kNearest:
            labels[index_kNearest] = self.train_labels[j]
            index_kNearest += 1
        
        return np.bincount(labels).argmax()

    def fit(self, training_data, training_labels):
        """
            Trains the model, returns predicted labels for training data.
            Hint: Since KNN does not really have parameters to train, you can try saving the training_data
            and training_labels as part of the class. This way, when you call the "predict" function
            with the test_data, you will have already stored the training_data and training_labels
            in the object.

            Arguments:
                training_data (np.array): training data of shape (N,D)
                training_labels (np.array): labels of shape (N,)
            Returns:
                pred_labels (np.array): labels of shape (N,)
        """

        ##
        ###
        #### YOUR CODE HERE!
        ###
        ##
        self.train_data = training_data
        self.train_labels = training_labels
        nbOfData = self.training_data.shape[1]
        pred_labels = np.zeros(nbOfData)
        for i in range(nbOfData):
            pred_labels[i] = self.label_find(self, training_data[i, :])
        
        return pred_labels

    def predict(self, test_data):
        """
            Runs prediction on the test data.

            Arguments:
                test_data (np.array): test data of shape (N,D)
            Returns:
                test_labels (np.array): labels of shape (N,)
        """
        ##
        ###
        #### YOUR CODE HERE!
        ###
        ##
        nbOfData = test_data.shape[1]
        test_labels = np.zeroes(nbOfData)
        for i in range(nbOfData):
            test_labels[i] = self.label_find(self, test_data[i, :])
            
        return test_labels