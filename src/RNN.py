import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class RNN:
    def __init__(self, random_state):
        # Create a new model containing 4 LSTM layers and 3 Dropout layers
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(64, return_sequences=True, time_major=True),
            tf.keras.layers.Dropout(.1),
            tf.keras.layers.LSTM(32, return_sequences=True, time_major=True),
            tf.keras.layers.Dropout(.1),
            tf.keras.layers.LSTM(16, return_sequences=True, time_major=True),
            tf.keras.layers.Dropout(.1),
            tf.keras.layers.LSTM(8, return_sequences=True, time_major=True),
            tf.keras.layers.Dense(4, activation='softmax')
        ])

        # Compile the model with ADAM optimizer and Categorical Cross Entropy Loss
        self.model.compile(optimizer=tf.keras.optimizers.Adam(),
                           loss=tf.keras.losses.CategoricalCrossentropy(),
                           metrics=[tf.keras.metrics.MeanAbsoluteError()])

    def encode_targets(self, y):
        targets = []
        for time in y:
            time = time.reshape(-1)
            time_arr = []
            # For each timestamp in each episode, one-hot encode the targets
            for instance in time:
                if instance == 'A':
                    time_arr.append([1, 0, 0, 0])
                elif instance == 'B':
                    time_arr.append([0, 1, 0, 0])
                elif instance == 'C':
                    time_arr.append([0, 0, 1, 0])
                else:
                    time_arr.append([0, 0, 0, 1])
            targets.append(time_arr)

        return np.array(targets)

    def fit(self, X, y, patience=2, epochs=100):
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                                          patience=patience,
                                                          mode='min')
        targets = self.encode_targets(y)

        # Fit the model to the training data
        history = self.model.fit(X, targets, epochs=epochs, callbacks=[early_stopping], validation_split=.1)

        # Graph the loss cruve of the model on the training data
        plt.plot(history.history['loss'])
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Recurrent Neural Network Training Loss')
        plt.savefig('img/rnn.png')

    def score(self, X):
        return self.model.predict(X)

    def predict(self, X):
        scores = self.score(X)

        predictions = []
        for time in scores:
            time_array = []
            # For each timestamp of each episode, convert the probability distribution to a prediction
            for score in time:
                pred = np.argmax(score)
                if pred == 0:
                    time_array.append('A')
                elif pred == 1:
                    time_array.append('B')
                elif pred == 2:
                    time_array.append('C')
                else:
                    time_array.append('D')

            predictions.append(time_array)

        return np.array(predictions)