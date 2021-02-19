import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import f1_score, confusion_matrix
import random

# Setting random seeds to keep everything deterministic.
random.seed(1618)
np.random.seed(1618)
# tf.set_random_seed(1618)   # Uncomment for TF1.
tf.random.set_seed(1618)

# Disable some troublesome logging.
# tf.logging.set_verbosity(tf.logging.ERROR)   # Uncomment for TF1.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Information on dataset.
NUM_CLASSES = 10
IMAGE_SIZE = 784

# Use these to set the algorithm to use.
# ALGORITHM = "guesser"
ALGORITHM = "custom_net"
# ALGORITHM = "tf_net"


class NeuralNetwork_2Layer():
    def __init__(self, inputSize, outputSize, neuronsPerLayer, learningRate=0.1):
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.neuronsPerLayer = neuronsPerLayer
        self.lr = learningRate
        self.W1 = np.random.randn(self.inputSize, self.neuronsPerLayer)
        self.W2 = np.random.randn(self.neuronsPerLayer, self.outputSize)

    # Activation function.
    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Activation prime function.
    def __sigmoidDerivative(self, x):
        return self.__sigmoid(x) * (1 - self.__sigmoid(x))

    # Batch generator for mini-batches. Not randomized.
    def __batchGenerator(self, l, n):
        i = 0
        length = len(l)
        while 1:
            if i + n > length - 1:
                i = 0
            yield l[i: i + n]
            i += n

    # Training with backpropagation.
    def train(self, xVals, yVals, epochs=100000, minibatches=True, mbs=100):
        batch_x = None
        batch_y = None
        if minibatches:
            batch_x = self.__batchGenerator(xVals, mbs)
            batch_y = self.__batchGenerator(yVals, mbs)
        else:
            batch_x = self.__batchGenerator(xVals, len(xVals))
            batch_y = self.__batchGenerator(yVals, len(yVals))

        for i in range(epochs):
            curr_x = next(batch_x)
            curr_y = next(batch_y)
            layer1, layer2 = self.__forward(curr_x)
            l2e = layer2 - curr_y
            l2d = l2e * self.__sigmoidDerivative(layer2)
            l1e = np.dot(l2d, self.W2.T)
            l1d = l1e * self.__sigmoidDerivative(layer1)
            l1a = np.dot(curr_x.T, l1d) * self.lr
            l2a = np.dot(layer1.T, l2d) * self.lr

            self.W1 -= l1a
            self.W2 -= l2a

    # Forward pass.
    def __forward(self, input):
        layer1 = self.__sigmoid(np.dot(input, self.W1))
        layer2 = self.__sigmoid(np.dot(layer1, self.W2))
        return layer1, layer2

    # Predict.
    def predict(self, xVals):
        _, layer2 = self.__forward(xVals)
        return layer2


# Classifier that just guesses the class label.
def guesserClassifier(xTest):
    ans = []
    for entry in xTest:
        pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        pred[random.randint(0, 9)] = 1
        ans.append(pred)
    return np.array(ans)


# =========================<Pipeline Functions>==================================

def getRawData():
    mnist = tf.keras.datasets.mnist
    (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    print("Shape of xTrain dataset: %s." % str(xTrain.shape))
    print("Shape of yTrain dataset: %s." % str(yTrain.shape))
    print("Shape of xTest dataset: %s." % str(xTest.shape))
    print("Shape of yTest dataset: %s." % str(yTest.shape))
    return ((xTrain, yTrain), (xTest, yTest))


def preprocessData(raw):
    ((xTrain, yTrain), (xTest, yTest)) = raw
    xTrain = xTrain.reshape(60000, IMAGE_SIZE)
    xTest = xTest.reshape(10000, IMAGE_SIZE)
    xTrain = xTrain / 255
    xTest = xTest / 255
    yTrainP = to_categorical(yTrain, NUM_CLASSES)
    yTestP = to_categorical(yTest, NUM_CLASSES)
    print("New shape of xTrain dataset: %s." % str(xTrain.shape))
    print("New shape of xTest dataset: %s." % str(xTest.shape))
    print("New shape of yTrain dataset: %s." % str(yTrainP.shape))
    print("New shape of yTest dataset: %s." % str(yTestP.shape))
    return ((xTrain, yTrainP), (xTest, yTestP))


def trainModel(data):
    xTrain, yTrain = data
    if ALGORITHM == "guesser":
        return None  # Guesser has no model, as it is just guessing.
    elif ALGORITHM == "custom_net":
        model = NeuralNetwork_2Layer(IMAGE_SIZE, NUM_CLASSES, 100)
        model.train(xTrain, yTrain, 500)
        return model
    elif ALGORITHM == "tf_net":
        model = keras.Sequential()
        model.add(tf.keras.layers.Dense(500, input_shape=(IMAGE_SIZE,), activation='relu'))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(500, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(10, activation='sigmoid'))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        single_label = np.argmax(yTrain, axis=1)
        model.fit(xTrain, single_label, epochs=10, verbose=0)
        return model
    else:
        raise ValueError("Algorithm not recognized.")


def runModel(data, model):
    if ALGORITHM == "guesser":
        return guesserClassifier(data)
    elif ALGORITHM == "custom_net":
        return model.predict(data)
    elif ALGORITHM == "tf_net":
        return model.predict(data)
    else:
        raise ValueError("Algorithm not recognized.")


def evalResults(data, preds):
    xTest, yTest = data
    acc = 0
    y_true = np.argmax(yTest, axis=1)
    y_pred = np.argmax(preds, axis=1)
    for i in range(preds.shape[0]):
        if y_pred[i] == y_true[i]:
            acc = acc + 1
    accuracy = acc / preds.shape[0]
    print("Classifier algorithm: %s \n" % ALGORITHM)
    print("Classifier confusion matrix: ")
    print(confusion_matrix(y_true, y_pred))
    print("Classifier F1 score (macro): " '{0:6f}'.format(f1_score(y_true, y_pred, average='macro')))
    print("Classifier accuracy: %f%%" % (accuracy * 100))


# =========================<Main>================================================

def main():
    raw = getRawData()
    data = preprocessData(raw)
    model = trainModel(data[0])
    preds = runModel(data[1][0], model)
    evalResults(data[1], preds)


if __name__ == '__main__':
    main()
