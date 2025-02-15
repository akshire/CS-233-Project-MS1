import argparse

import numpy as np

from src.data import load_data
from src.methods.dummy_methods import DummyClassifier
from src.methods.logistic_regression import LogisticRegression
from src.methods.linear_regression import LinearRegression 
from src.methods.knn import KNN
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, mse_fn
import time
import os
np.random.seed(100)


def main(args):
    """
    The main function of the script. Do not hesitate to play with it
    and add your own code, visualization, prints, etc!

    Arguments:
        args (Namespace): arguments that were parsed from the command line (see at the end 
                          of this file). Their value can be accessed as "args.argument".
    """
    ## 1. First, we load our data and flatten the images into vectors

    ##EXTRACTED FEATURES DATASET
    if args.data_type == "features":
        feature_data = np.load('features.npz',allow_pickle=True)
        xtrain, xtest, ytrain, ytest, ctrain, ctest =feature_data['xtrain'],feature_data['xtest'],\
        feature_data['ytrain'],feature_data['ytest'],feature_data['ctrain'],feature_data['ctest']

    ##ORIGINAL IMAGE DATASET (MS2)
    elif args.data_type == "original":
        data_dir = os.path.join(args.data_path,'dog-small-64')
        xtrain, xtest, ytrain, ytest, ctrain, ctest = load_data(data_dir)

    ##TODO: ctrain and ctest are for regression task. (To be used for Linear Regression and KNN)
    ##TODO: xtrain, xtest, ytrain, ytest are for classification task. (To be used for Logistic Regression and KNN)

    ## 2. Then we must prepare it. This is were you can create a validation set,
    #  normalize, add bias, etc.
    

    # Make a validation set (it can overwrite xtest, ytest)
    if not args.test:
        ### WRITE YOUR CODE HERE
        fraction_train = 0.7

        
        test_random_indexes = np.random.permutation(xtrain.shape[0])
        
        n_train = int(xtrain.shape[0]*fraction_train)
        xdata = xtrain
        ydata = ytrain
        
        xtrain = xdata[test_random_indexes[:n_train]]
        ytrain = ydata[test_random_indexes[:n_train]] 
        
        xtest = xdata[test_random_indexes[n_train:]] 
        ytest = ydata[test_random_indexes[n_train:]] 

        ctest = ctrain[test_random_indexes[n_train:]] 
        ctrain = ctrain[test_random_indexes[:n_train]] 

        
        pass
    
    ### WRITE YOUR CODE HERE to do any other data processing
    
    xtest = normalize_fn(xtest, xtrain.mean(0,keepdims=True), xtrain.std(0,keepdims=True))
    xtest = append_bias_term(xtest)

    xtrain = normalize_fn(xtrain, xtrain.mean(0,keepdims=True), xtrain.std(0,keepdims=True))
    xtrain = append_bias_term(xtrain)
    
    


    ## 3. Initialize the method you want to use.

    # Use NN (FOR MS2!)
    if args.method == "nn":
        raise NotImplementedError("This will be useful for MS2.")

    # Follow the "DummyClassifier" example for your methods
    if args.method == "dummy_classifier":
        method_obj = DummyClassifier(arg1=1, arg2=2)

    elif args.method == "linear_regression": 
        method_obj = LinearRegression(lmda = args.lmda)
        pass

    elif args.method == "logistic_regression":
        method_obj = LogisticRegression(lr = args.lr, max_iters = args.max_iters)
        pass

    elif args.method == "knn":
        if args.task == "center_locating":
            task_kind = 'regression'
        elif args.task == "breed_identifying":
            task_kind = 'classification'
        help = args.kNN_help
        help_parameters = np.array([args.Kfold_values_folds,args.Kfold_values_start,args.Kfold_values_end,args.Kfold_values_spacing])

        method_obj = KNN(k = args.K, task_kind = task_kind,help=help,help_parameters=help_parameters)
        pass


    ## 4. Train and evaluate the method

    if args.task == "center_locating":
        # Fit parameters on training data
        zero = time.time()
        preds_train = method_obj.fit(xtrain, ctrain)
        fit = time.time()
        # Perform inference for training and test data
        train_pred = method_obj.predict(xtrain)
        preds = method_obj.predict(xtest)
        pred = time.time()
        ## Report results: performance on train and valid/test sets
        print(f"Time to fit {fit-zero:.3f} seconds \nTime to predict {pred-fit:.3f} seconds")
        train_loss = mse_fn(train_pred, ctrain)
        loss = mse_fn(preds, ctest)

        print(f"\nTrain loss = {train_loss:.3f}% - Test loss = {loss:.3f}")

    elif args.task == "breed_identifying":

        # Fit (:=train) the method on the training data for classification task
        zero = time.time()
        preds_train = method_obj.fit(xtrain, ytrain)
        fit = time.time()
        # Predict on unseen data
        preds = method_obj.predict(xtest)
        pred = time.time()
        ## Report results: performance on train and valid/test sets
        acc = accuracy_fn(preds_train, ytrain)
        macrof1 = macrof1_fn(preds_train, ytrain)
        print(f"Time to fit {fit-zero:.3f} seconds \nTime to predict {pred-fit:.3f} seconds")
        print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

        acc = accuracy_fn(preds, ytest)
        macrof1 = macrof1_fn(preds, ytest)
        print(f"Test set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")
    else:
        raise Exception("Invalid choice of task! Only support center_locating and breed_identifying!")

    ### WRITE YOUR CODE HERE if you want to add other outputs, visualization, etc.


if __name__ == '__main__':
    # Definition of the arguments that can be given through the command line (terminal).
    # If an argument is not given, it will take its default value as defined below.
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default="center_locating", type=str, help="center_locating / breed_identifying")
    parser.add_argument('--method', default="dummy_classifier", type=str, help="dummy_classifier / knn / linear_regression/ logistic_regression / nn (MS2)")
    parser.add_argument('--data_path', default="data", type=str, help="path to your dataset")
    parser.add_argument('--data_type', default="features", type=str, help="features/original(MS2)")
    parser.add_argument('--lmda', type=float, default=1, help="lambda of linear/ridge regression")
    parser.add_argument('--K', type=int, default=20, help="number of neighboring datapoints used for knn")
    parser.add_argument('--lr', type=float, default=2.02, help="learning rate for methods with learning rate")
    parser.add_argument('--max_iters', type=int, default=1228, help="max iters for methods which are iterative")
    parser.add_argument('--test', action="store_true", help="train on whole training data and evaluate on the test data, otherwise use a validation set")


    # Feel free to add more arguments here if you need!
    parser.add_argument('--kNN_help', action="store_true", help="For kNN classification, should the K-fold method be used to print the best k (much slower)")
    parser.add_argument('--Kfold_values_folds',type=int, default = 5,help="Give the amount of folds K for K-fold")
    parser.add_argument('--Kfold_values_start',type=int, default = 4,help="Give the starting k for K-fold")
    parser.add_argument('--Kfold_values_end',type=int, default = 40,help="Give the end k for K-fold")
    parser.add_argument('--Kfold_values_spacing',type=int, default = 4,help="Give the space between tries of k for K-fold")
    # MS2 arguments
    parser.add_argument('--nn_type', default="cnn", help="which network to use, can be 'Transformer' or 'cnn'")
    parser.add_argument('--nn_batch_size', type=int, default=64, help="batch size for NN training")

    # "args" will keep in memory the arguments and their values,
    # which can be accessed as "args.data", for example.
    args = parser.parse_args()
    main(args)