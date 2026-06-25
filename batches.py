"""
    ===============================================================================================================================
    *
    * The first row of inputs must match the number of columns in weights for the dot product to be valid.
    
    **Think of it this way:
                        ir * |ic|
                             |wr| * wc  #Where r = rows, c = columns and i = inputs, w = weights
    * The output will have ir * wc elements


    * In numbers:
                    (3 , 4)    ****    (4 ,  3  ) =  (3 , 3)
                     ^   ^              ^    ^
                     |   4 ============ 4    |
                     |                       |
                     3 samples               3 neurons

                     

    * Biases (br,) are broadcasted across all rows of the output:
                        ir * |  ic  |
                             |  wr  | * wc  +  br
    * In numbers:
                        (3,3) + (3,) = (3,3)
    * Each bias belongs to a neuron, not a sample — every sample gets the same bias shift per neuron
    ===============================================================================================================================
"""

import numpy as np

#"================================================================"
#layer 1 inputs, weights, and biases
inputs = [
    [1,2,3,2.5],
    [2.0,5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]
]

weights = [
    [0.2, 0.8, -0.5, 1.0], 
    [0.5, -0.91, 0.26, -0.5], 
    [-0.26, -0.27, 0.17, 0.87]
]


biases = [2, 3, 0.5]

layer1_outputs = np.dot(inputs, np.array(weights).T) + biases
print("layer 1 outputs: ")
print(layer1_outputs)

#"================================================================"

#Layer 2 weights, and biases
#Now layer 2 will take the outputs of layer 1 as its inputs, and we will define new weights and biases for layer 2.
weights2 = [
    [0.1, -0.14, 0.5], 
    [-0.5, 0.12, -0.33],
    [-0.44, 0.73, -0.13]
]

biases2 = [-1, 2, -0.5]

layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2
print("\n" + "layer 2 outputs: ")
print(layer2_outputs)