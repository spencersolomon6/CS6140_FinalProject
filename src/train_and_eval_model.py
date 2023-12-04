import sys
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from LogisticRegression import LogRegression
from NN import NeuralNetwork
from RNN import RNN
from GaussianProcess import GaussianProcess
from sklearn import metrics
from imblearn.under_sampling import RandomUnderSampler
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic
from joblib import dump, load
import keras_tuner as kt
import tensorflow as tf
import matplotlib.pyplot as plt

RANDOM_STATE = 11


def extract_X_y(data, columns, classes, nsamples):
    '''
    Extract the relevant features and targets from the given data

    :param data: the DataFrame to extract from
    :param columns: the relevant feature columns
    :param classes: the relevant target column
    :param nsamples: The number of samples to extract
    :return: features, targets
    '''
    scaler = StandardScaler()
    continuous_columns = ['eeg_fp1', 'eeg_f7',
                          'eeg_f8', 'eeg_t4', 'eeg_t6', 'eeg_t5', 'eeg_t3', 'eeg_fp2', 'eeg_o1',
                          'eeg_p3', 'eeg_pz', 'eeg_f3', 'eeg_fz', 'eeg_f4', 'eeg_c4', 'eeg_p4',
                          'eeg_poz', 'eeg_c3', 'eeg_cz', 'eeg_o2']

    # First, we split the data up by crew, dividing into continuous timeseries for each experiment
    data_dict = {}
    max_length = 0
    for crew in data['crew'].unique():
        # Extract each episode and reorder it by time
        timeseries = data[data['crew'] == crew]
        timeseries = timeseries.sort_values(by=['time'], ignore_index=True)

        # If nsamples was passed, take that many samples. Otherwise, use all the data
        if nsamples is not None:
            data_dict[int(crew)] = timeseries.sample(nsamples)
            max_length = nsamples
        else:
            data_dict[int(crew)] = timeseries
            max_length = len(timeseries) if len(timeseries) > max_length else max_length

    # For each timeseries episode, extract the features and targets
    inputs = []
    targets = []
    for crew, samples in data_dict.items():
        X = samples[columns]
        y = samples[classes]

        # Since each of the episodes must be the same length, be pad the data with the means for each column and the
        # baseline target state to reach that length
        means = pd.Series({col: X[col].mean() for col in X.columns})
        X = X.reindex(range(max_length)).fillna(means)
        y = y.reindex(range(max_length)).fillna('A')

        # Remove outliers in the data via mean imputation
        for col in continuous_columns:
            threshold1 = X[col].quantile(.95)
            threshold2 = X[col].quantile(.05)
            mean = X[col].mean()
            X.loc[:, col] = [val if threshold1 >= val >= threshold2 else mean for val in
                             X[col]]

        '''
        # Used when testing PCA on the Logistic Regression Classifier 
        pca = PCA(n_components=1)
        X = pca.fit_transform(X.to_numpy())
        '''

        # Normalize the input data before adding it to the final array
        inputs.append(scaler.fit_transform(X)[None, :])
        targets.append(y)

    # Combine each episode into a 3D array of shape (episodes, timestamps, features)
    return np.vstack(inputs).astype('float32'), np.array(targets)


def undersample(inputs, targets):
    '''
    Undersample the data using a Random UnderSampler

    :param inputs:
    :param targets:
    :return: resampled_inputs, resampled_targets
    '''
    sampler = RandomUnderSampler()

    final_inputs, final_targets = [], []
    resampled = []
    max_length = 0
    for crew_inputs, crew_targets in zip(inputs, targets):
        # For each episode in the data, undersample the targets and inputs so that all classes have the same probability
        crew_targets = [target[0] for target in crew_targets]
        y = pd.get_dummies(crew_targets).to_numpy()

        # Resample the data
        X, y = sampler.fit_resample(crew_inputs, y)

        # Reconstruct the targets array
        resampled_targets = [np.unique(crew_targets)[np.argmax(row)] for row in y]
        max_length = len(resampled_targets) if len(resampled_targets) > max_length else max_length

        resampled.append((X, resampled_targets))

    for input, target in resampled:
        final_inputs.append(pd.DataFrame(input).reindex(range(max_length)).fillna(0))
        final_targets.append(pd.DataFrame(target).reindex(range(max_length)).fillna('A'))

    return np.vstack(final_inputs).astype('float32'), np.vstack(final_targets)


def clean_preprocess(data, test_split, nsamples=None, resample=False):
    '''
    Given a DataFrame of pilot data, this method extracts the features and targets, normalizes the data, splits the data
    into train and test sets and formats the inputs as a dictionary of continuous timeseries keyed by their crew number

    :param data: The dataset to clean/preprocess
    :param test_split: The proportion of data to keep as a test set
    :param nsamples: The max number of samples per episode (Optional)
    :param resample: Whether to undersample the data or not (Optional)
    :return: (train_X, train_y, test_X, test_y)
    '''
    columns = ['crew', 'time', 'seat', 'eeg_fp1', 'eeg_f7',
       'eeg_f8', 'eeg_t4', 'eeg_t6', 'eeg_t5', 'eeg_t3', 'eeg_fp2', 'eeg_o1',
       'eeg_p3', 'eeg_pz', 'eeg_f3', 'eeg_fz', 'eeg_f4', 'eeg_c4', 'eeg_p4',
       'eeg_poz', 'eeg_c3', 'eeg_cz', 'eeg_o2', 'ecg', 'r', 'gsr']
    classes = ['event']

    # Randomize the order of the data before extracting the test set
    data = data.sample(frac=1, random_state=RANDOM_STATE, ignore_index=True)

    test_set_size = int(len(data) * test_split)
    train, test = data[test_set_size:], data[:test_set_size]

    # Extract the features and targets from the train and test data
    train_X, train_y = extract_X_y(train, columns, classes, nsamples)
    test_X, test_y = extract_X_y(test, columns, classes, nsamples)

    if resample:
        train_X, train_y = undersample(train_X, train_y)

    return train_X, train_y, test_X, test_y


def evaluate_model(X, y, model):
    '''
    Calculate evaluation metrics for the given model

    :param X: The feature test data
    :param y: The target test data
    :param model: The model to evaluate
    :return: accuracy, fscore, class_accuracies
    '''
    # Make predictions on the feature data and unfold the time column of both arrays
    predictions = model.predict(X).reshape(-1)
    targets = y.reshape(-1)

    # Create a confusion matrix of the results
    cm = metrics.confusion_matrix(targets, predictions)

    # Calculate the metrics
    accuracy = metrics.accuracy_score(targets, predictions)
    class_accuracies = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    class_accuracies = class_accuracies.diagonal()
    fscore = metrics.f1_score(targets, predictions, average='weighted')


    return accuracy, fscore, class_accuracies


def train_eval_logistic_regression(train_X, train_y, test_X, test_y):
    '''
    Train and evaluate the Logistic Regression Model

    :param train_X: Training features
    :param train_y: Training targets
    :param test_X: Testing features
    :param test_y: Testing targets
    :return: Evaluation metrics
    '''
    lr = LogRegression(RANDOM_STATE)
    lr.fit(train_X, train_y)
    scores = evaluate_model(test_X, test_y, lr)

    dump(lr, '../models/lr.joblib')

    return scores


def train_eval_gaussian_process(train_X, train_y, test_X, test_y, kernel=None):
    '''
    Train and evaluate the Gaussian Process Model

    :param train_X: Training features
    :param train_y: Training targets
    :param test_X: Testing features
    :param test_y: Testing targets
    :return: Evaluation metrics
    '''
    gp = GaussianProcess(RANDOM_STATE, kernel)
    gp.fit(train_X, train_y)
    scores = evaluate_model(test_X, test_y, gp)

    dump(gp, '../models/gp.joblib')

    return scores


def train_eval_recurrent_neural_network(train_X, train_y, test_X, test_y):
    '''
    Train and evaluate the Recurrent Neural Network Model

    :param train_X: Training features
    :param train_y: Training targets
    :param test_X: Testing features
    :param test_y: Testing targets
    :return: Evaluation metrics
    '''
    rnn = RNN(RANDOM_STATE)
    rnn.fit(train_X, train_y, epochs=50)
    scores = evaluate_model(test_X, test_y, rnn)

    dump(rnn, '../models/rnn.joblib')

    return scores


def train_eval_neural_network(train_X, train_y, test_X, test_y):
    '''
    Train and evaluate the Deep Neural Network Model

    :param train_X: Training features
    :param train_y: Training targets
    :param test_X: Testing features
    :param test_y: Testing targets
    :return: Evaluation metrics
    '''
    nn = NeuralNetwork(RANDOM_STATE)
    nn.fit(train_X, train_y, epochs=50)
    scores = evaluate_model(test_X, test_y, nn)

    dump(nn, '../models/nn.joblib')

    return scores


def tune_nn_hyperparameters(X, y):
    '''

    :param X:
    :param y:
    :return:
    '''

    nn = NeuralNetwork()
    tuner = kt.Hyperband(nn.build_hyperparam_model,
                         objective='val_accuracy',
                         max_epochs=20,
                         factor=3,
                         directory='hyperparam_search',
                         project_name='DNN'
                         )

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    features = X.reshape(-1, X.shape[-1])
    targets = nn.encode_targets(y.reshape(-1))

    tuner.search(features, targets, epochs=50, validation_split=.2, callbacks=[stop_early])
    best_settings = tuner.get_best_hyperparameters(num_trials=1)[0]

    units = best_settings.get('units')
    activation = best_settings.get('activation')
    lr = best_settings.get('learning_rate')
    dropout = best_settings.get('dropout')

    print(f'Optimal Hyperparameter Settings:\n'
          f'Units of FC Layer: {units}\n'
          f'Activation Function: {activation}\n'
          f'Dropout Rate: {dropout}\n'
          f'Learning Rate: {lr}')


def print_scores(scores, model_name):
    '''
    Print the evaluation results

    :param scores: evaluation results
    :param model_name: the name of the current model
    :return: None
    '''
    print(f'{model_name} Scores --\n Accuracy: {scores[0]}, F-Score: {scores[1]}, Class Accuracies: {scores[2]}')


def load_eval_model(model_name, X, y):
    '''
    Load and evaluate a model from a file

    :param model_name: The abbreviated name of the model
    :param X: Test features
    :param y: Test targets
    :return: None
    '''
    model = load(f'models/{model_name}.joblib')
    print_scores(evaluate_model(X, y, model), model_name)

    return model


def find_interesting_errors(model, X, y, should_print=True):
    loss = tf.keras.losses.categorical_crossentropy

    inputs = X.reshape(-1, X.shape[-1])
    targets = model.encode_targets(y.reshape(-1))

    losses = []
    for prediction, target, input in zip(model.score(inputs), targets, inputs):
        losses.append((loss(target, prediction), input, prediction, target))

    top_errors = sorted(losses, key=lambda item: -item[0])[:5]

    if should_print:
        print(f'Errors:\n')
        for loss, input, prediction, target in top_errors:
            print(f'Loss: {loss}')
            print(f'Inputs: {input}')
            print(f'Prediction: {prediction}')
            print(f'Target: {target}\n')

    return top_errors


def main():
    # Read in which type of model should be run
    if len(sys.argv) > 1:
        model_type = sys.argv[1]
    else:
        model_type = 'help'

    if len(sys.argv) > 2 and str(sys.argv[2]) == '1':
        retrain = True
    else:
        retrain = False

    # Read in and clean/process the training data
    train = pd.read_csv('../data/balanced_data.csv')
    train_X, train_y, test_X, test_y = clean_preprocess(train, test_split=.2, nsamples=None, resample=False)

    # For each model type: If the model file exists, load and evaluate it. Otherwise, train and evaluate a new model on
    # the data
    if model_type == 'lr':
        if not retrain and os.path.exists('../models/lr.joblib'):
            load_eval_model('lr', test_X, test_y)
        else:
            scores = train_eval_logistic_regression(train_X, train_y, test_X, test_y)
            print_scores(scores, 'Logistic Regression')
    elif model_type == 'gp':
        if not retrain and os.path.exists('../models/gp.joblib'):
            load_eval_model('gp', test_X, test_y)
        else:
            train_X, train_y, test_X, test_y = clean_preprocess(train, test_split=.2, nsamples=1000, resample=False)

            kernels = {'Rational Quadratic': RationalQuadratic(1.0), 'RBF': RBF(1.0)}
            for kernel_name, kernel in kernels.items():
                scores = train_eval_gaussian_process(train_X, train_y, test_X, test_y, kernel)
                print(f'Gaussian Process trained on {kernel_name} kernel')
                print_scores(scores, 'Gaussian Process')
    elif model_type == 'rnn':
        if not retrain and os.path.exists('../models/rnn.joblib'):
            load_eval_model('rnn', test_X, test_y)
        else:
            scores = train_eval_recurrent_neural_network(train_X, train_y, test_X, test_y)
            print_scores(scores, 'Recurrent Neural Network')
    elif model_type == 'nn':
        if not retrain and os.path.exists('../models/nn.joblib'):
            model = load_eval_model('nn', test_X, test_y)
            errors = find_interesting_errors(model, test_X, test_y, should_print=False)
        else:
            tune_nn_hyperparameters(train_X, train_y)
            scores = train_eval_neural_network(train_X, train_y, test_X, test_y)
            print_scores(scores, 'Neural Network')
    else:
        print(f'Please specify a model type to train/test:\n'
              f'* lr - Logistic Regression\n'
              f'* gp - Gaussian Process\n'
              f'* rnn - Recurrent Neural Network\n\n'
              f'EX: python src/train_and_eval_model.py lr\n\n'
              f'If you would like to retrain the models instead of using the pretrained files, set the second arg to 1.\n\n'
              f'EX: python src/train_and_eval_model.py nn 1')


if __name__ == '__main__':
    main()