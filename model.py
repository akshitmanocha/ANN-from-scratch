import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import ReLU
from utils import SoftMax
from utils import ReLU_deriv
from utils import one_hot
from dataset import X_train,Y_train,X_val,Y_val

class ANN_from_scratch():
    def __init__(self):
        # Initializing weights
        self.W1 = np.random.rand(224, 784) - 0.5
        self.b1 = np.random.rand(224, 1) - 0.5
        self.W2 = np.random.rand(10, 224) - 0.5
        self.b2 = np.random.rand(10, 1) - 0.5
        
        # Adam optimizer parameters
        self.mW1, self.vW1 = np.zeros_like(self.W1), np.zeros_like(self.W1)
        self.mb1, self.vb1 = np.zeros_like(self.b1), np.zeros_like(self.b1)
        self.mW2, self.vW2 = np.zeros_like(self.W2), np.zeros_like(self.W2)
        self.mb2, self.vb2 = np.zeros_like(self.b2), np.zeros_like(self.b2)
        self.epsilon = 1e-8
        
    def forward(self, X):
        self.Z1 = self.W1.dot(X) + self.b1
        self.A1 = ReLU(self.Z1)
        self.Z2 = self.W2.dot(self.A1) + self.b2
        self.A2 = SoftMax(self.Z2)
        return self.A2
    
    def backprop(self, Y, X):
        m = Y.size
        self.one_hot_Y = one_hot(Y)
        self.dZ2 = self.A2 - self.one_hot_Y
        self.dW2 = 1 / m * self.dZ2.dot(self.A1.T)
        self.db2 = 1 / m * np.sum(self.dZ2, axis=1, keepdims=True) 
        self.dZ1 = self.W2.T.dot(self.dZ2) * ReLU_deriv(self.Z1)
        self.dW1 = 1 / m * self.dZ1.dot(X.T)
        self.db1 = 1 / m * np.sum(self.dZ1, axis=1, keepdims=True)
        
    def get_predictions(self, A2):
        return np.argmax(A2, axis=0)
    
    def learning_rate_decay(self, lr, t, decay_rate):
        self.lr = lr * (1 / (1 + t * decay_rate))
        
    def get_accuracy(self, predictions, Y):
        return np.sum(predictions == Y) / Y.size
    
    def adam_optimizer(self, param, grad, m, v, beta1, beta2, lr, t):
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        param -= lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return param, m, v
    
    def update_param(self, lr, beta1=0.9, beta2=0.999):
        self.W1, self.mW1, self.vW1 = self.adam_optimizer(self.W1, self.dW1, self.mW1, self.vW1, beta1, beta2, lr, self.iteration)
        self.b1, self.mb1, self.vb1 = self.adam_optimizer(self.b1, self.db1, self.mb1, self.vb1, beta1, beta2, lr, self.iteration)
        self.W2, self.mW2, self.vW2 = self.adam_optimizer(self.W2, self.dW2, self.mW2, self.vW2, beta1, beta2, lr, self.iteration)
        self.b2, self.mb2, self.vb2 = self.adam_optimizer(self.b2, self.db2, self.mb2, self.vb2, beta1, beta2, lr, self.iteration)
        
    def train(self, iterations, X, Y, lr):
        self.iteration = 1
        for i in range(iterations):
            A2 = self.forward(X)
            self.backprop(Y, X)
            self.update_param(lr)
            
            if i % 10 == 0:
                self.train_predictions = self.get_predictions(A2)
                print(f'Iteration: {i} \nTrain Accuracy: {int(self.get_accuracy(self.train_predictions, Y) * 100)}%')
                self.check_accuracy(X_val.T,Y_val)
            self.iteration += 1
    
    def check_accuracy(self, X, Y):
        A2 = self.forward(X)
        test_predictions = self.get_predictions(A2)
        print(f'Test Accuracy: {int(self.get_accuracy(test_predictions, Y) * 100)}%\n')
    
    def make_predictions(self, X):
        A2 = self.forward(X)
        predictions = self.get_predictions(A2)
        return predictions

    def test_prediction(self, index, X_train, Y_train):
        current_image = X_train[:, index, None]
        prediction = self.make_predictions(current_image)
        label = Y_train[index]
        print("Prediction: ", prediction)
        print("Label: ", label)

        current_image = current_image.reshape((28, 28)) * 255
        plt.gray()
        plt.imshow(current_image, interpolation='nearest')
        plt.show()

if __name__ == '__main__':
    model = ANN_from_scratch()
    model.train(100,X_train.T,Y_train,0.01)