from NeuralNetworkClasses import NeuralNetwork, Layer, Neuron
import numpy as np

x_train = [[[0,0]], [[0,1]], [[1,0]], [[1,1]]]
y_train = [[[0]], [[1]], [[1]], [[0]]]

testLayerNeuronCount = [2, 3, 1]
testLayerTypes = ["ReLU", "ReLU", "sigmoid"]

test = NeuralNetwork(2, testLayerNeuronCount, testLayerTypes)
test.train(x_train, y_train, 10000)
