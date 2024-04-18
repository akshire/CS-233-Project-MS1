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
        """
            Compute the Euclidean distance between a single example
            vector and all training_examples.

            Inputs:
                example: shape (,D)
                training_examples: shape (N, D) 
            Outputs:
                euclidean distances: shape (N,)
        """
        
        return np.sqrt(((training_examples - example) ** 2).sum(axis=1))

    def computeKNearest(self, test_point, index, bool):
        """
            Compute the labels of the k nearest neighbors

            Arguments:
                test_point (np.array): point that we want to label (,D)
                index (not used): position of the test_point in the original array
                bool: boolean value to determin if we are either in fit (= 0) or prediction(= 1)
            Outputs:
                labels: labels of the k nearest neighbors (k,)
        """
        
        euclid_distances = self.euclidean_dist(test_point, np.delete(self.train_data, index, 0) if (not bool) else self.train_data)
        kNearest = np.argpartition(euclid_distances, self.k)[:self.k]
        labels = np.zeros(self.k)
        index_kNearest = 0
        if (not bool):
            labelsWithoutTestedPoint = np.delete(self.train_labels, index, 0)
            for j in kNearest:
                labels[index_kNearest] = labelsWithoutTestedPoint[j]
                index_kNearest += 1
        else:
            for j in kNearest:
                labels[index_kNearest] = self.train_labels[j]
                index_kNearest += 1
                
        return labels

    def label_find(self, test_point, index, bool):
        """
            Return the predicted label by calculating all the distances between the point and all the 
            training_data and then returning the most frequent one from k nearest neighbors

            Arguments:
                test_point (np.array): point that we want to label (,D)
                index (not used): position of the test_point in the original array
                bool: boolean value to determin if we are either in fit (= 0) or prediction(= 1)
            Outputs:
                predicted label: label of the shape
        """

        kNearestLabels = self.computeKNearest(test_point, index, bool)
        if (self.task_kind == "classification"):
            uniqueLabels, labelsCountFrequency = np.unique(kNearestLabels, return_counts = True)
            return uniqueLabels[labelsCountFrequency.argmax()]
        else:
            return np.mean(labels)

        
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
        nbOfData = training_data.shape[0]
        pred_labels = np.zeros(nbOfData)

        for i in range(nbOfData):
            pred_labels[i] = self.label_find(training_data[i, :], i, 0)
        
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
        nbOfData = test_data.shape[0]
        test_labels = np.zeros(nbOfData)
        
        for i in range(nbOfData):
            test_labels[i] = self.label_find(test_data[i, :], i, 1)
            
        return test_labels