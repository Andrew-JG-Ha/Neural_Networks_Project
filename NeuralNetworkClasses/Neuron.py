import numpy as np
import random
import math

class Neuron():
    """
    Neuron class
    - the lowest level of a neural network
    - contains a structure and 
    """
    def __init__(self, numberOfWeights:int, seed = None) -> None:
        np.random.seed(seed)
        self.weights = generateWeights(numberOfWeights, seed)
        self.bias = 0 # np.random.random()*0.1
        self.neuronInput = []
        self.neuronValue = None
        self.neuronOutput = None
        self.neuronPrime = None

    def updateWeights(self, newWeights:list) -> None:
        for weightIndex in range(0, len(newWeights)):
            self.weights[weightIndex] = newWeights[weightIndex]
    def updateBias(self, newBias) -> None:
        self.bias = newBias
    def updateNeuronInput(self, neuronInput) -> None:
        self.neuronInput = neuronInput
    def updateNeuronValue(self, neuronValue) -> None:
        self.neuronValue = neuronValue
    def updateNeuronOutput(self, newNeuronOutput) -> None:
        self.neuronOutput = newNeuronOutput

    def getNeuronOutput(self):
        return self.neuronOutput
    def getNeuronValue(self):
        return self.neuronValue
    def getWeights(self):
        return self.weights
    def getBias(self):
        return self.bias
    def getPrimeValue(self):
        return self.neuronPrime

    def activate(self, activationType) -> None:
        if activationType == "Step":
            self.neuronOutput = unitStep(self.neuronValue)
        elif activationType == "Sigmoid":
            self.neuronOutput = sigmoid(self.neuronValue)
        elif activationType == "Tanh":
            self.neuronOutput = tanh(self.neuronValue)
        elif activationType == "ReLu":
            self.neuronOutput = relu(self.neuronValue)

    def activatePrime(self, activationType) -> None:
        if activationType == "Step":
            self.neuronPrime = unitStepPrime(self.neuronValue)
        elif activationType == "Sigmoid":
            self.neuronPrime = sigmoidPrime(self.neuronValue)
        elif activationType == "Tanh":
            self.neuronPrime = tanhPrime(self.neuronValue)
        elif activationType == "ReLu":
            self.neuronPrime = reluPrime(self.neuronValue)

def generateWeights(numberOfWeights, seed = None) -> list:
    output = list()
    np.random.seed(seed)
    for i in range(0, numberOfWeights):
        output.append(np.random.random()*0.1)
    return output

"""
activation functions
"""
def unitStep(input):
    if (input < 0):
        return 0
    else:
        return 1
def sigmoid(input):
    activated = (1/(1+np.exp(-input)))
    return activated
def tanh(input):
    activated = (np.exp(2*input) - 1) / (np.exp(2*input) + 1)
    return activated
def relu(input, alpha = 0.01):
    if (input < 0):
        return alpha*(np.exp(input) - 1)
    else:
        return input

"""
activation derivative functions
"""
def unitStepPrime(input):
    if (input == 0):
        return 1
    else:
        return 0
def sigmoidPrime(input):
    activated = (np.exp(-input)) / pow(1+np.exp(-input), 2)
    return activated
def tanhPrime(input):
    activated = (4*np.exp(-2*input)) / pow((1 + np.exp(-2*input)), 2)
    return activated
def reluPrime(input, alpha = 0.01):
    if (input < 0):
        return alpha * np.exp(input)
    else:
        return 1

# test that the Neuron is working as intended:
# def testNeuron():
#     test = Neuron(4, 99)
#     print(test.getNeuronOutput)
#     print(test.getNeuronValue())
#     print(test.getWeights)

# testNeuron()