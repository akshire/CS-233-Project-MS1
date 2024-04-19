import numpy as np

class KNN(object):
    """
        kNN classifier object.
    """

    def __init__(self, k=1, task_kind = "regression"):
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

    def computeKNearest(self, test_point):
        """
            Compute the labels of the k nearest neighbors

            Arguments:
                test_point (np.array): point that we want to label (,D)
                index (not used): position of the test_point in the original array
                bool: boolean value to determin if we are either in fit (= 0) or prediction(= 1)
            Outputs:
                labels: labels of the k nearest neighbors (k,)
        """
        euclid_distances = self.euclidean_dist(test_point,self.train_data)
        kNearest_labels = self.train_labels[np.argpartition(euclid_distances, self.k)[:self.k]]
        #TODO add weight values        
        
                
        return kNearest_labels

    def label_find(self, test_point):
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

        if (self.task_kind == "classification"):
            euclid_distances = self.euclidean_dist(test_point,self.train_data)
            indexes = np.argpartition(euclid_distances, self.k)[:self.k]
            kNearest_labels = self.train_labels[indexes]
            euclid_distances_nearest = euclid_distances[indexes]
            weights = 1/euclid_distances_nearest



            nbr_label_of_each_type = np.bincount(kNearest_labels,weights)
            return nbr_label_of_each_type.argmax()
        else:
            euclid_distances = self.euclidean_dist(test_point,self.train_data)
            kNearest_points = self.train_labels[np.argpartition(euclid_distances, self.k)[:self.k]]
            return np.mean(kNearest_points,axis=0)

        
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
        print("shape output ", training_labels.shape)
        pred_labels = np.zeros(training_labels.shape)
        #TODO could add a helper to choose k

        for i in range(nbOfData):
            pred_labels[i] = self.label_find(training_data[i, :])
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
        size_of_outputs = len(self.train_labels.shape)
        if size_of_outputs < 2:
            test_labels = np.zeros([nbOfData])
        else:
            size_of_outputs = self.train_labels.shape[1]
            test_labels = np.zeros([nbOfData,size_of_outputs])
        for i in range(nbOfData):
            test_labels[i] = self.label_find(test_data[i, :])
        return test_labels