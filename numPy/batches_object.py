
import numpy as np

np.random.seed(0) # for reproducibility

#X -- Input data
X = [
    [1,2,3,2.5],
    [2.0,5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]
]

#Hidden layer 1 -- forward pass
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        #Initialize weights and biases
        #we are trying to avoid transpose on forward pass, so we will initialize weights as (n_inputs, n_neurons) instead of (n_neurons, n_inputs) 
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons) #random weights with small values
        self.biases = np.zeros((1, n_neurons)) #biases initialized to zero

    def forward(self, inputs):
        #Calculate output values from inputs, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases


layer1 = Layer_Dense(4,5) #4 inputs, 5 neurons
layer1.forward(X)
print("layer 1 outputs: ")
print(layer1.output)

layer2 = Layer_Dense(5,2) #5 inputs, 2 neurons 
layer2.forward(layer1.output)
print("\n" + "layer 2 outputs: ")
print(layer2.output)






# weights = [
#     [0.2, 0.8, -0.5, 1.0], 
#     [0.5, -0.91, 0.26, -0.5], 
#     [-0.26, -0.27, 0.17, 0.87]
# ]


# biases = [2, 3, 0.5]

# layer1_outputs = np.dot(inputs, np.array(weights).T) + biases
# print("layer 1 outputs: ")
# print(layer1_outputs)

# #"================================================================"

# #Layer 2 weights, and biases
# #Now layer 2 will take the outputs of layer 1 as its inputs, and we will define new weights and biases for layer 2.
# weights2 = [
#     [0.1, -0.14, 0.5], 
#     [-0.5, 0.12, -0.33],
#     [-0.44, 0.73, -0.13]
# ]

# biases2 = [-1, 2, -0.5]

# layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2
# print("\n" + "layer 2 outputs: ")
# print(layer2_outputs)