from keras.datasets import mnist
from keras.utils import to_categorical
from keras import Sequential, Input, layers, optimizers
from sklearn.model_selection import train_test_split 
import numpy as np

class AnnKeras: 
  def __init__(self, activation, learning_rate, epochs, batch_size, ratio_train, arquitecture): 
    # hyperparameters
    self.activation = activation
    self.learning_rate = learning_rate
    self.epochs = epochs
    self.batch_size = batch_size
    # params
    self.ratio_train = ratio_train
    # model
    self.arquitecture = arquitecture.split(',')
    self.model = self.get_model()
    self.history_loss = []
    self.history_acc = []
    self.test_loss = 0
    self.test_acc = 0

  def load_data(self): 
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype(np.float32) / 255
    x_test = x_test.astype(np.float32) / 255

    X = np.vstack((x_train, x_test))
    y = np.hstack((y_train, y_test))
    X = X.reshape(70000, 784)
    y = to_categorical(y, num_classes = 10)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1 - self.ratio_train, stratify = y)
    return X_train, X_test, y_train, y_test

  def get_model(self): 
    model = Sequential()

    for idx, neurons in enumerate(self.arquitecture): 
      if idx == 0: 
        model.add(Input(shape = (int(neurons), )))
      elif idx == len(self.arquitecture) - 1: 
        model.add(layers.Dense(int(neurons), activation = 'softmax'))
      else: 
        model.add(layers.Dense(int(neurons), activation = self.activation))

    model.compile(
      loss = 'categorical_crossentropy',
      optimizer = optimizers.SGD(learning_rate = self.learning_rate), 
      metrics = ['accuracy']
    )
    return model

  def train(self): 
    X_train, X_test, y_train, y_test = self.load_data()
    history = self.model.fit(X_train, y_train, batch_size = self.batch_size, epochs = self.epochs, verbose = 0)
    test_loss, test_acc = self.model.evaluate(X_test, y_test)

    self.history_loss = history.history['loss']
    self.history_acc = history.history['accuracy']
    self.test_loss = test_loss
    self.test_acc = test_acc

  def getHistoryLoss(self): 
    return self.history_loss
  
  def getHistoryAcc(self): 
    return self.history_acc