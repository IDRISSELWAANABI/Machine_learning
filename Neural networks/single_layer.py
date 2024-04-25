import numpy as np 
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def backward(self, X, y, output):
        error = output - y
        delta_output = error * self.sigmoid_derivative(output)
        error_hidden = delta_output.dot(self.W2.T)
        delta_hidden = error_hidden * self.sigmoid_derivative(self.a1)
        self.W2 -= self.a1.T.dot(delta_output)
        self.b2 -= np.sum(delta_output, axis=0, keepdims=True)
        self.W1 -= X.T.dot(delta_hidden)
        self.b1 -= np.sum(delta_hidden, axis=0, keepdims=True)

    def train(self, X, y, epochs):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)
            if epoch % 1000 == 0:
                loss = np.mean(np.square(y - output))
                print(f'Epoch {epoch}: Loss = {loss:.4f}')


