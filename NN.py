import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, random_state=11, optimal=True):
        # Create a DNN with 4 Dense layers and 3 Dropout layers
        if not optimal:
            self.model = tf.keras.models.Sequential([
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dropout(.1),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dropout(.1),
                tf.keras.layers.Dense(16, activation='relu'),
                tf.keras.layers.Dropout(.1),
                tf.keras.layers.Dense(8, activation='relu'),
                tf.keras.layers.Dense(4, activation='softmax')
            ])
            # Compile the model with the ADAM optimizer and Categorical Cross Entropy loss
            self.model.compile(optimizer=tf.keras.optimizers.Adam(),
                               loss=tf.keras.losses.CategoricalCrossentropy(),
                               metrics=[tf.keras.metrics.MeanAbsoluteError()])
        else:
            self.model = tf.keras.models.Sequential([
                tf.keras.layers.Dense(128, 'sigmoid'),
                tf.keras.layers.Dropout(.01),
                tf.keras.layers.Dense(4, 'softmax')
            ])
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=.01),
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=[tf.keras.metrics.MeanAbsoluteError()]
            )

    def encode_targets(self, y):
        targets = []
        # For each timestamp, one-hot encode the targets
        for instance in y:
            if instance == 'A':
                targets.append([1, 0, 0, 0])
            elif instance == 'B':
                targets.append([0, 1, 0, 0])
            elif instance == 'C':
                targets.append([0, 0, 1, 0])
            else:
                targets.append([0, 0, 0, 1])

        return np.array(targets)

    def fit(self, X, y, patience=2, epochs=100):
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                                          patience=patience,
                                                          mode='min')
        inputs = X.reshape(-1, X.shape[-1])
        targets = y.reshape(-1)

        targets = self.encode_targets(targets)

        history = self.model.fit(inputs, targets, epochs=epochs, callbacks=[early_stopping], validation_split=.1)

        plt.plot(history.history['loss'])
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Deep Neural Network Training Loss')
        plt.savefig('img/nn.png')

    def score(self, X):
        return self.model.predict(X)

    def predict(self, X):
        inputs = X.reshape(-1, X.shape[-1])
        scores = self.score(inputs)

        predictions = []
        for score in scores:
            pred = np.argmax(score)
            if pred == 0:
                predictions.append('A')
            elif pred == 1:
                predictions.append('B')
            elif pred == 2:
                predictions.append('C')
            else:
                predictions.append('D')

        return np.array(predictions)

    def build_hyperparam_model(self, hp):
        '''
        Generate a hyper model definition to perform hyperparameter turing for the DNN

        :param hp:
        :return:
        '''

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(26))

        hp_units = hp.Int('units', min_value=16, max_value=256, step=32)
        hp_activation = hp.Choice('activation', values=['relu', 'tanh', 'sigmoid'])
        model.add(keras.layers.Dense(units=hp_units, activation=hp_activation))

        hp_dropout = hp.Choice('dropout', values=[.01, .1, .2])
        model.add(keras.layers.Dropout(hp_dropout))

        model.add(keras.layers.Dense(4, activation='softmax'))

        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                      loss=keras.losses.CategoricalCrossentropy(),
                      metrics=['accuracy'])

        return model