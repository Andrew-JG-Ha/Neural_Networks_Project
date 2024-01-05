import numpy as np
import random
import math

class Neuron():

    def __init__(self, numberOfWeights:int, seed = None) -> None:
        self.neuronInput = []
        self.neuronOutput = None
        np.random.seed(seed)
        self.weights = generateWeights(numberOfWeights, seed)
        self.bias = np.random.random()*0.1
        self.neuronValue = None
        self.neuronPrime = None

    def updateWeights(self, newWeights:list) -> None:
        for weightIndex in range(0, len(newWeights)):
            self.weights[weightIndex] = newWeights[weightIndex]

    def updateBias(self, newBias) -> None:
        self.bias = newBias

    def updateNeuronValue(self, neuronValue) -> None:
        self.neuronValue = neuronValue

    def getNeuronValue(self):
        return self.neuronValue
    
    def getWeights(self):
        return self.weights
    
    def getBias(self):
        return self.bias
    
    def getPrimeValue(self):
        return self.neuronPrime

    def updateNeuronInput(self, neuronInput) -> None:
        self.neuronInput = neuronInput
        weightedSum = 0
        for input, weight in zip(self.neuronInput, self.weights):
            weightedSum = weightedSum + input * weight
        weightedSum = weightedSum + self.bias
        self.neuronOutput = weightedSum

    def getNeuronOutput(self):
        return self.neuronOutput

    def activate(self, activationType) -> None:
        if activationType == "Step":
            self.neuronOutput = unitStep(self.neuronValue)
            self.neuronPrime = unitStepPrime(self.neuronValue)
        elif activationType == "Sigmoid":
            self.neuronOutput = sigmoid(self.neuronValue)
            self.neuronPrime = sigmoidPrime(self.neuronValue)
        elif activationType == "Tanh":
            self.neuronOutput = tanh(self.neuronValue)
            self.neuronPrime = tanhPrime(self.neuronValue)
        elif activationType == "ReLu":
            self.neuronOutput = relu(self.neuronValue)
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