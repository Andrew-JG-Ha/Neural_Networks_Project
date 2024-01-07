from random import *
import numpy as np

"""
Hard coded neural network with 5 neurons in a  2-1 setup
"""
class hardcodedNN_singleOut:
    def __init__(self, learningRate) -> None:
        self.learningRate = learningRate
        self.weight1 = np.random.random()*0.1
        self.weight2 = np.random.random()*0.1
        self.weight3 = np.random.random()*0.1
        self.weight4 = np.random.random()*0.1
        self.weight5 = np.random.random()*0.1
        self.weight6 = np.random.random()*0.1

        self.bias1 = np.random.random()*0.1
        self.bias2 = np.random.random()*0.1
        self.bias3 = np.random.random()*0.1

        self.neuron1H = 0
        self.neuron2H = 0
        self.neuron3H = 0

        self.neuron1AH = 0
        self.neuron2AH = 0
        self.neuron3AH = 0

        self.input = None
        self.output = None

    def forwardPropagate(self, inputs:list):
        # Layer 1
        self.input = inputs
        self.neuron1H = self.weight1 * inputs[0] + self.weight2 * inputs[1] + self.bias1
        self.neuron2H = self.weight3 * inputs[0] + self.weight4 * inputs[1] + self.bias2
        self.neuron1AH = sigmoid(self.neuron1H)
        self.neuron1_dAct_dPred = sigmoidPrime(self.neuron1H)
        self.neuron2AH = sigmoid(self.neuron2H)
        self.neuron2_dAct_dPred = sigmoidPrime(self.neuron2H)

        # Layer 2
        self.neuron3H = self.weight5 * self.neuron1AH + self.weight6 * self.neuron2AH + self.bias3
        self.neuron3AH = sigmoid(self.neuron3H)
        self.neuron3_dAct_dPred = sigmoidPrime(self.neuron3H)

        # Pass layer 2 output to network output
        self.output = self.neuron3AH
        return self.output
    
    def backPropagate(self, expectedValue, actualValue):
        dLoss_dPred = meanSquaredErrorPrime([expectedValue], [actualValue])[0]

        dPred_dW6 = self.neuron3_dAct_dPred * self.neuron2AH
        dPred_dW5 = self.neuron3_dAct_dPred * self.neuron1AH
        dPred_dB3 = self.neuron3_dAct_dPred * 1

        dPred_dH2 = self.neuron3_dAct_dPred * self.weight6
        dPred_dH1 = self.neuron3_dAct_dPred * self.weight5

        dH2_dW4 = self.neuron2_dAct_dPred * self.input[1]
        dH2_dW3 = self.neuron2_dAct_dPred * self.input[0]
        dH2_dB2 = self.neuron2_dAct_dPred * 1

        dH1_dW2 = self.neuron1_dAct_dPred * self.input[1]
        dH1_dW1 = self.neuron1_dAct_dPred * self.input[0]
        dH1_dB1 = self.neuron1_dAct_dPred * 1

        # update the weights
        self.weight6 = self.weight6 - self.learningRate * dLoss_dPred * dPred_dW6
        self.weight5 = self.weight5 - self.learningRate * dLoss_dPred * dPred_dW5
        self.weight4 = self.weight4 - self.learningRate * dLoss_dPred * dPred_dH2 * dH2_dW4
        self.weight3 = self.weight3 - self.learningRate * dLoss_dPred * dPred_dH2 * dH2_dW3
        self.weight2 = self.weight2 - self.learningRate * dLoss_dPred * dPred_dH1 * dH1_dW2
        self.weight1 = self.weight1 - self.learningRate * dLoss_dPred * dPred_dH1 * dH1_dW1

        #update biases
        self.bias3 = self.bias3 - self.learningRate * dLoss_dPred * dPred_dB3
        self.bias2 = self.bias2 - self.learningRate * dLoss_dPred * dPred_dH2 * dH2_dB2
        self.bias1 = self.bias1 - self.learningRate * dLoss_dPred * dPred_dH1 * dH1_dB1

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
    
"""
error functions
"""
def meanSquaredError(expectedValues:list, actualValues:list):
    return (1/len(expectedValues))*pow(sum([(expected-actual) for expected, actual in zip(expectedValues, actualValues)]), 2)

def meanSquaredErrorPrime(expectedValues:list, actualValues:list):
    return ([-(2/len(expectedValues))*(expected-actual) for expected, actual in zip(expectedValues, actualValues)])

def meanAbsoluteError(expectedValues:list, actualValues:list):
    return (1/len(expectedValues)) * sum([abs(expected-actual) for expected, actual in zip(expectedValues, actualValues)])

def meanAbsoluteErrorPrime(expectedValues:list, actualValues:list):
    return [1 if actual > expected else -1 for expected, actual in zip(expectedValues, actualValues)]


def test():
    testNN = hardcodedNN_singleOut(0.05)

    trainingData = [[-2, -1], [25, 6], [17, 4], [17, 4], [-15, -6]]
    correctAnswers = [1, 0, 0, 1]

    for epoch in range(0, 1000):
        for data, correctAnswer in zip(trainingData, correctAnswers):
            result = testNN.forwardPropagate(data)
            testNN.backPropagate(correctAnswer, result)
    result = testNN.forwardPropagate([-2, -1])
    result = testNN.forwardPropagate([25, 6])
    result = testNN.forwardPropagate([17, 4])
    result = testNN.forwardPropagate([-15, -6])

test()