import numpy as np
import random
import math

class Neuron():
    """
    Neuron class
    """
    def __init__(self, numberOfWeights:int, neuronType:str, layerPosition:int, seed = 105) -> None:
        np.random.seed(seed)
        self.weights = generateWeights(numberOfWeights)
        self.bias = np.random.random()*0.1
        self.neuronType = neuronType
        self.layerPosition = layerPosition
        self.neuronValue = None
        self.neuronOutput = None

    def updateWeight(self, neuronNumber:int, newValue) -> None:
        self.weights[neuronNumber] = newValue

    def addWeight(self, weightValue) -> None:
        self.weights[len(self.weights) + 1] = weightValue

    def updateBias(self, newValue) -> None:
        self.bias = newValue
    
    def calculateNeuronValue(self, inputs:list) -> None:
        "Calculate the value of all the weighted inputs to this neuron"
        if len(inputs) != len(self.weights):
            print("The input array is not the correct length")
            raise ValueError("Input array is not the correct length")
        else:
            result = np.dot(self.weights, inputs) + self.bias
            self.neuronValue = result

    def calculateNeuronOutput(self, inputs:list, alpha = 1):
        "Apply an activation function"
        self.calculateNeuronValue(inputs)
        x = self.neuronValue
        result = None
        if (self.neuronType == "sigmoid"):
            result = sigmoid(x)
            self.neuronOutput = result
        if (self.neuronType == "tanh"):
            result = tanh(x)
            self.neuronOutput = result
        elif (self.neuronType == "ReLU"):
            result = ReLU(x)
            self.neuronOutput = result
        elif (self.neuronType == "leaky ReLU"):
            result = leakyReLU(x)
            self.neuronOutput = result
        elif (self.neuronType == "ELU"):
            result = ELU(x, alpha)
            self.neuronOutput = result
        return result
    
    def getNeuronValue(self):
        return self.neuronValue

    def getNeuronOutput(self):
        return self.neuronOutput

def generateWeights(numberOfWeights) -> dict:
    output = list()
    for i in range(0, numberOfWeights):
        output.append(np.random.random()*0.1)
    return output

def sigmoid(x):
    return 1/(1+np.exp(-x))

def ReLU(x):
    return np.maximum(0, x)

def leakyReLU(x):
    return np.maximum(0.1*x, x)

def tanh(x):
    return math.tanh(x)

def ELU(x, alpha = 1):
    if x >= 0:
        return x
    else:
        return alpha*(np.exp(x)-1)

test = Neuron(3, "test", 1)
test.calculateNeuronValue([1,2,3])
print("test")

# each neuron will require as many weights as there are neurons in the previous layer

# there can only be 1 bias per layer