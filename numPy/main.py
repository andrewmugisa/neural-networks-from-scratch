"""
    ===============================================================================================================================

    * This is a simple implementation of a neural network layer using a manual loop and NumPy for computation.
    * It demonstrates how to compute the outputs of a layer of neurons given inputs, weights, and biases.

    **Think of it this way:
                            outputs = inputs * weights + biases
    same as:                y = mx + b

    * The code first computes the outputs using a manual loop and then compares it with the NumPy implementation using the dot product.
    * The outputs from both methods should be the same, confirming the correctness of the manual implementation
    * and the efficiency of using NumPy for such operations.
    *
    * The inputs represent the features fed into the layer, while the weights and biases are parameters that
    * determine how the inputs are transformed into outputs. Each neuron in the layer has its own set of weights and a bias, 
    * and the output of each neuron is computed as the weighted sum of the inputs plus the bias.
    *
    * The final outputs of the layer are printed to the console, allowing for verification of the results.
    *
    * Note: The code assumes that the inputs, weights, and biases are correctly shaped and compatible for the operations performed.

    ===============================================================================================================================
""" 



#import sys
import numpy as np

#print("Python version:", sys.version) #3.13.13 
#print("NumPy version:", np.__version__) #2.5.0


print('------------------------------------------------------------------------------')
inputs = [1,2,3,2.5] #we usually dont control the inputs, but can tune the weights and biases to get the desired outputs

#(3,4) 3 neurons, 4 inputs
neuron_weights = [[0.2, 0.8, -0.5, 1.0], [0.5, -0.91, 0.26, -0.5], [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]  # one bias per neuron, shifts the output independent of inputs

#using a manual loop to compute the outputs of the layer
layer_outputs = []
for neuron_w, neuron_bias in zip(neuron_weights, biases):
    neuron_output = 0
    for n_input, weight in zip(inputs, neuron_w):
        neuron_output += n_input * weight
    neuron_output += neuron_bias
    layer_outputs.append(neuron_output)
print("loop=== " + str(layer_outputs))

print()

#using NumPy to compute the outputs of the layer
numpy_layer_outputs = np.dot(neuron_weights, inputs) + biases
print("NumPy=== " + str(numpy_layer_outputs))


print('------------------------------------------------------------------------------')




#a simple diagram to visualize the layer of neurons and their connections to the inputs
print("""
  x1 (1.0)   ─┐
  x2 (2.0)   ─┼──► N1
  x3 (3.0)   ─┤
  x4 (2.5)   ─┘

  x1 (1.0)   ─┐
  x2 (2.0)   ─┼──► N2
  x3 (3.0)   ─┤
  x4 (2.5)   ─┘

  x1 (1.0)   ─┐
  x2 (2.0)   ─┼──► N3
  x3 (3.0)   ─┤
  x4 (2.5)   ─┘
""")

# for i, neuron_w in enumerate(neuron_weights):
#     for j, val in enumerate(inputs):
#         connector = '─┼──►' if j == len(inputs) // 2 else ('─┐' if j == 0 else ('─┘' if j == len(inputs)-1 else '─┤'))
#         print(f"  x{j+1} ({val})   {connector}" + (f" N{i+1}" if j == len(inputs) // 2 else ""))
#     print()