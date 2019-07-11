import numpy as np
import matplotlib.pyplot as plt

"""Neuronal Network class definition"""
class NeuralNetwork:

    # initialise the neural network
    def __init__(self, inputs, hidden, outputs, lr):
        # set number of neurons in each input, hidden and output layer and learning rate
        self.inputs = inputs
        self.hidden = hidden
        self.outputs = outputs
        self.lr = lr

        # activation function is sigmoid function
        self.activation = lambda x: 1/(1+np.exp(-x))

        # Define weight matrices from input to hidden layer wih and hidden to output layer who
        self.wih = np.random.normal(0.0, pow(self.hidden, -0.5), (self.hidden, self.inputs))
        self.who = np.random.normal(0.0, pow(self.outputs, -0.5), (self.outputs, self.hidden))

    # train the neural network
    def train(self, inputs_list, targets_list):
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        # forward calculations for hidden and output layer
        hidden_in = np.dot(self.wih, inputs)
        hidden_out = self.activation(hidden_in)

        final_in = np.dot(self.who, hidden_out)
        final_out = self.activation(final_in)  # final function = sigmoid = classification

        # backpropagation
        out_err = targets - final_out
        # error in hidden layer is computed by multiplying with weights
        hidden_err = np.dot(self.who.T, out_err)

        # update the weights. This is the gradient descent part for the sigmoid activation
        self.who += self.lr * np.dot((out_err * final_out * (1.0 - final_out)), hidden_out.T)
        self.wih += self.lr * np.dot((hidden_err * hidden_out * (1.0 - hidden_out)), inputs.T)

    # Forward the signal neural network
    def predict(self, input_list):
        # convert list into 2d numpy array
        inputs = np.array(input_list, ndmin=2).T

        # hidden layer calculations
        hidden_in = np.dot(self.wih, inputs)
        hidden_out = self.activation(hidden_in)

        # final output layer calculations
        final_in = np.dot(self.who, hidden_out)
        final_out = self.activation(final_in)  # final function = sigmoid = classification

        return final_out


# https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


"""Load and shuffle the data"""
# load data 2
train_in = np.loadtxt("data/mnist_small_train_in2.csv", skiprows=1, delimiter=";")
train_out = np.loadtxt("data/mnist_small_train_out.csv")
# shuffle data
train_in, train_out = unison_shuffled_copies(train_in, train_out)

test_in = np.loadtxt("data/mnist_small_test_in2.csv", skiprows=1, delimiter=";")
test_out = np.loadtxt("data/mnist_small_test_out.csv")

test_in, test_out = unison_shuffled_copies(test_in, test_out)



"""Set the Hyperparameter for the Network """
input_nodes = 784  # 28x28 = 784
hidden_nodes = 200  # My Machine Learning sense tells me this is a good number
output_nodes = 10  # We have 10 labels

lr = 0.15
epochs = 15


# create instance of NN
model = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, lr)


error_epoch = []
for e in range(epochs):
    """Train the Neural Network """
    for i in range(len(train_in)):
        #inputs = (train_in[i,:] / 255.0 * 0.99) + 0.01  # We would need this for the actualy mnist data which I used before this.
        inputs = train_in[i,:]

        targets = np.zeros(output_nodes) + 0.01  # Add 0.01 for numerical stability
        targets[int(train_out[i])] = 0.99  # One hot encoding. Take the current value from train_out as index for targets

        model.train(inputs, targets)  # Magic time

    """Validate the results"""
    score = []
    for i in range(len(test_in)):
        correct_label = test_out[i]

        #inputs = (train_in[i, :] / 255.0 * 0.99) + 0.01  # Again no need for in this mnist version.
        inputs = test_in[i,:]

        outputs = model.predict(inputs)

        label = np.argmax(outputs)

        # Keep track of the score
        if (label == correct_label):
            score.append(1)
        else:
            score.append(0)

    accuracy = (np.asarray(score)).sum() / (np.asarray(score)).size
    error = (1.0 - accuracy) * 100
    print("ERROR: ", round(error, 2), "%")

    error_epoch.append(error)

"""Plot the results"""
plt.rcParams.update({'font.size': 36})
plt.plot(error_epoch)
plt.xlabel("Epoch count")
plt.ylabel("Error in %")
plt.show()


