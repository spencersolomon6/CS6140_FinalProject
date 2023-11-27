import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from LogisticRegression import LogRegression
from RNN import RNN
from GaussianProcess import GaussianProcess
from sklearn import metrics

RANDOM_STATE = 11

def extract_X_y(data, columns, classes):
    '''

    :param data:
    :return:
    '''
    scaler = StandardScaler()

    data_dict = {}
    max_length = 0
    for crew in data['crew'].unique():
        timeseries = data[data['crew'] == crew]
        timeseries = timeseries.sort_values(by=['time'], ignore_index=True)
        # timeseries = timeseries.drop(['time', 'crew'], axis=1)

        data_dict[int(crew)] = timeseries

        max_length = len(timeseries) if len(timeseries) > max_length else max_length

    inputs = []
    targets = []
    for crew, samples in data_dict.items():
        X = samples[columns]
        y = samples[classes]
        X = X.reindex(range(max_length)).fillna(0)
        y = y.reindex(range(max_length)).fillna('A')

        inputs.append(scaler.fit_transform(X.to_numpy().astype('float32'))[None, :])
        targets.append(y.to_numpy())

    return np.vstack(inputs), np.array(targets)



def clean_preprocess(data, test_split):
    '''
    Given a DataFrame of pilot data, this method extracts the features and targets, normalizes the data, splits the data
    into train and test sets and formats the inputs as a dictionary of continous timeseries keyed by their crew number

    :param data:
    :param test_split:
    :return:
    '''
    columns = ['crew', 'time', 'seat', 'eeg_fp1', 'eeg_f7',
       'eeg_f8', 'eeg_t4', 'eeg_t6', 'eeg_t5', 'eeg_t3', 'eeg_fp2', 'eeg_o1',
       'eeg_p3', 'eeg_pz', 'eeg_f3', 'eeg_fz', 'eeg_f4', 'eeg_c4', 'eeg_p4',
       'eeg_poz', 'eeg_c3', 'eeg_cz', 'eeg_o2', 'ecg', 'r', 'gsr']
    classes = ['event']

    data = data.sample(frac=1, random_state=RANDOM_STATE, ignore_index=True)

    # targets = pd.get_dummies(data['event'])
    # data = pd.concat([data, targets], axis=1)

    test_set_size = int(len(data) * test_split)
    train, test = data[test_set_size:], data[:test_set_size]

    train_X, train_y = extract_X_y(train, columns, classes)
    test_X, test_y = extract_X_y(test, columns, classes)

    return train_X, train_y, test_X, test_y

def evaluate_model(X, y, model):
    predictions = model.predict(X).reshape(-1)
    targets = y.reshape(-1)

    accuracy = metrics.accuracy_score(targets, predictions)
    fscore = metrics.f1_score(targets, predictions, average='weighted')
    kappa = metrics.cohen_kappa_score(targets, predictions)

    return accuracy, fscore, kappa

def train_eval_logistic_regression(train_X, train_y, test_X, test_y):
    '''

    :param train_X:
    :param train_y:
    :param test_X:
    :param test_y:
    :return:
    '''
    lr = LogRegression(RANDOM_STATE)
    lr.fit(train_X, train_y)
    scores = evaluate_model(test_X, test_y, lr)

    return scores

def train_eval_gaussian_process(train_X, train_y, test_X, test_y):
    '''

    :param train_X:
    :param train_y:
    :param test_X:
    :param test_y:
    :return:
    '''
    gp = GaussianProcess(RANDOM_STATE)
    gp.fit(train_X, train_y)
    scores = evaluate_model(test_X, test_y, gp)

    return scores


def train_eval_recurrent_neural_network(train_X, train_y, test_X, test_y):
    '''

    :param train_X:
    :param train_y:
    :param test_X:
    :param test_y:
    :return:
    '''
    rnn = RNN(RANDOM_STATE)
    rnn.fit(train_X, train_y)
    scores = evaluate_model(test_X, test_y, rnn)

    return scores

def print_scores(scores, model_name):
    print(f'{model_name} Scores --\n Accuracy: {scores[0]}, F-Score: {scores[1]}, Kappa Score: {scores[2]}')

def main():
    # Read in and clean/process the training data
    train = pd.read_csv('data/.train.csv')

    train_X, train_y, test_X, test_y = clean_preprocess(train, test_split=.2)

    scores = train_eval_logistic_regression(train_X, train_y, test_X, test_y)
    print_scores(scores, 'Logistic Regression')

    #scores = train_eval_gaussian_process(train_X, train_y, test_X, test_y)
    #print_scores(scores, 'Gaussian Process')

    scores = train_eval_recurrent_neural_network(train_X, train_y, test_X, test_y)
    print_scores(scores, 'Recurrent Neural Network')


if __name__ == '__main__':
    main()