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
        euclid_distances = self.euclidean_dist(test_point,self.train_data)
        #gets indexes of the k nearest samples
        indexes = np.argpartition(euclid_distances, self.k)[:self.k]
        kNearest_labels = self.train_labels[indexes]
        euclid_distances_nearest = euclid_distances[indexes]
        if (self.task_kind == "classification"):
            #set a minimum distance value to avoid division by zero problems
            euclid_distances_nearest[euclid_distances_nearest<0.001] = 0.001
            weights = 1/euclid_distances_nearest
            nbr_label_of_each_type = np.bincount(kNearest_labels,weights)
            return nbr_label_of_each_type.argmax()
        elif (self.task_kind == "regression"):
            kNearest_points = self.train_labels[indexes]
            #set a minimum distance value to avoid division by zero problems
            euclid_distances_nearest[euclid_distances_nearest<0.001] = 0.001
            weights = 1/euclid_distances_nearest
            return np.average(kNearest_points,weights=weights,axis=0)
        else:
            print("Invalid task_kind input")
            return

        
    def fit(self, training_data, training_labels,help = True):
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
        self.train_data = training_data
        self.train_labels = training_labels
        nbOfData = training_data.shape[0]
        pred_labels = np.zeros(training_labels.shape)
        def Kfold(folds=5,start=1,spacing=5,end=100):
            k_list = range(start,end,spacing)
            best_k = 0 # only store best value
            k_acc = 0
            for k in k_list:
                N = self.train_data.shape[0]
                accuracies = []  # list of accuracies
                # randomise the data picked
                random_indexes = np.random.permutation(self.train_data.shape[0])
                split_size = N // folds
                for fold in range(folds):
                    # get test and train indexes
                    test_indexes = random_indexes[fold * split_size : (fold + 1) * split_size]
                    train_indexes = np.setdiff1d(random_indexes,test_indexes)
                    # create new train/test np.arrays
                    train_data = self.train_data[train_indexes, :]
                    train_labels = self.train_labels[train_indexes]
                    test_data = self.train_data[test_indexes, :]
                    test_labels = self.train_labels[test_indexes]
                    # do knn and save accuracy to the accuracy list of this fold
                    method_obj = KNN(k = k, task_kind = self.task_kind)
                    method_obj.fit(train_data,train_labels,False)
                    pred = method_obj.predict(test_data)
                    acc = np.size(test_labels[pred==test_labels])/np.size(test_labels)
                    accuracies.append(acc)
                # compute the average accuracy of this fold
                ave_acc = np.sum(accuracies)/np.size(accuracies)
                if (ave_acc > k_acc):
                    k_acc = ave_acc
                    best_k = k
            print(f"The best k between {start} and {end} with a spacing of {spacing} is {best_k}")
            print("It has an accuracy of :", k_acc)

        if (help & (self.task_kind == "classification")):
            Kfold(spacing=1,end=30)



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