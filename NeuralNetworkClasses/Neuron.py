import numpy as np
import random
import math

class Neuron():
    """
    Neuron class
    """
    def __init__(self, numberOfWeights:int, seed = 105) -> None:
        np.random.seed(seed)
        self.weights = generateWeights(numberOfWeights)
        self.bias = np.random.random()*0.1
        self.neuronValue = None
        self.neuronOutput = None

    def updateAllWeights(self, newWeights:list) -> None:
        self.weights = newWeights

    def updateBias(self, newBias) -> None:
        self.bias = newBias

    def getNeuronValue(self):
        return self.neuronValue
    
    def updateNeuronValue(self, newNeuronValue):
        self.neuronValue = newNeuronValue

    def getNeuronOutput(self):
        return self.neuronOutput
    
    def updateNeuronOutput(self, newNeuronOutput):
        self.neuronOutput = newNeuronOutput

    def getWeights(self):
        return self.weights

def generateWeights(numberOfWeights) -> list:
    output = list()
    for i in range(0, numberOfWeights):
        output.append(np.random.random()*0.1)
    return output

# test that the Neuron is working as intended:
def testNeuron():
    test = Neuron(4, 0, 99)