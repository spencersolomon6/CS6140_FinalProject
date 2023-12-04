# CS6140 Final Project Group 20

## How to run the program

First, unzip the pretrained models until a folder `./models`, so that the models can 
be evaluated without requiring the user to wait for the long training process to complete.

train_and_eval_model.py is the main file of the program and accepts two parameters:

1. Model Name (Required)
    * Determines which model will be evaluated
      * lr - Logistic Regression
      * gp - Gaussian Process
      * rnn - Recurrent Neural Network
      * nn - Deep Neural Network
3. Retrain (Optional)
   * Overrides the default and retrains the model if set to 1

## Examples

To train and evaluate the Gaussian Process model without retraining:
```commandline
python train_eval_model.py gp
```

To train and evaluate the Deep Neural Network model with retraining:
```commandline
python train_eval_model.py rnn 1
```