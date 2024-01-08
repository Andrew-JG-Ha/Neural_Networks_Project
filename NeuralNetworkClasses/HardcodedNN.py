from random import *
import numpy as np

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


"""
Hard coded neural network with 3 neurons in a  2-1 setup
"""
class hardcodedNN_twoIn_singleOut:
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
Hard coded neural network with 6 neurons in a 3-2-1 setup --> 3 inputs, 1 outputs
"""
class hardcodedNN_threeIn_singleOut:
    def __init__(self, learningRate) -> None:
        self.learningRate = learningRate
        self.weight1 = np.random.random()*0.1
        self.weight2 = np.random.random()*0.1
        self.weight3 = np.random.random()*0.1
        self.weight4 = np.random.random()*0.1
        self.weight5 = np.random.random()*0.1
        self.weight6 = np.random.random()*0.1
        self.weight7 = np.random.random()*0.1
        self.weight8 = np.random.random()*0.1
        self.weight9 = np.random.random()*0.1

        self.weight10 = np.random.random()*0.1
        self.weight11 = np.random.random()*0.1
        self.weight12 = np.random.random()*0.1
        self.weight13 = np.random.random()*0.1
        self.weight14 = np.random.random()*0.1
        self.weight15 = np.random.random()*0.1

        self.weight16 = np.random.random()*0.1
        self.weight17 = np.random.random()*0.1

        self.bias1 = np.random.random()*0.1
        self.bias2 = np.random.random()*0.1
        self.bias3 = np.random.random()*0.1
        self.bias4 = np.random.random()*0.1
        self.bias5 = np.random.random()*0.1
        self.bias6 = np.random.random()*0.1

        self.neuron1H = 0
        self.neuron2H = 0
        self.neuron3H = 0
        self.neuron4H = 0
        self.neuron5H = 0
        self.neuron6H = 0

        self.neuron1AH = 0
        self.neuron2AH = 0
        self.neuron3AH = 0
        self.neuron4AH = 0
        self.neuron5AH = 0
        self.neuron6AH = 0

        self.input = None
        self.output = None

    def forwardPropagate(self, inputs:list):
        # Layer 1
        self.input = inputs
        self.neuron1H = self.weight1 * inputs[0] + self.weight2 * inputs[1] + self.weight3 * inputs[2] + self.bias1
        self.neuron2H = self.weight4 * inputs[0] + self.weight5 * inputs[1] + self.weight6 * inputs[2] + self.bias2
        self.neuron3H = self.weight7 * inputs[0] + self.weight8 * inputs[1] + self.weight9 * inputs[2] + self.bias3

        self.neuron1AH = sigmoid(self.neuron1H)
        self.neuron1_dAct_dH1 = sigmoidPrime(self.neuron1H)
        self.neuron2AH = sigmoid(self.neuron2H)
        self.neuron2_dAct_dH2 = sigmoidPrime(self.neuron2H)
        self.neuron3AH = sigmoid(self.neuron3H)
        self.neuron3_dAct_dH3 = sigmoidPrime(self.neuron3H)

        # Layer 2
        self.neuron4H = self.weight10 * self.neuron1AH + self.weight11 * self.neuron2AH + self.weight12 * self.neuron3AH + self.bias4
        self.neuron4AH = sigmoid(self.neuron4H)
        self.neuron4_dAct_dH4 = sigmoidPrime(self.neuron4H)
        self.neuron5H = self.weight13 * self.neuron1AH + self.weight14 * self.neuron2AH + self.weight15 * self.neuron3AH + self.bias5
        self.neuron5AH = sigmoid(self.neuron5H)
        self.neuron5_dAct_dH5 = sigmoidPrime(self.neuron4H)

        # Layer 3
        self.neuron6H = self.weight16 * self.neuron4AH + self.weight17 * self.neuron5AH + self.bias6
        self.neuron6AH = sigmoid(self.neuron6H)
        self.neuron6_dAct_dH6 = sigmoidPrime(self.neuron6AH)

        # Pass layer 3 output to network output
        self.output = self.neuron6AH
        return self.output
    
    def backPropagate(self, expectedValue, actualValue):
        dLoss_dH6 = meanSquaredErrorPrime([expectedValue], [actualValue])[0]

        dH6_dW17 = self.neuron6_dAct_dH6 * self.neuron5AH
        dH6_dW16 = self.neuron6_dAct_dH6 * self.neuron4AH
        dH6_dB6 = self.neuron6_dAct_dH6 * 1

        dH6_dH5 = self.neuron6_dAct_dH6 * self.weight17
        dH6_dH4 = self.neuron6_dAct_dH6 * self.weight16

        dH5_dW15 = self.neuron5_dAct_dH5 * self.neuron3AH
        dH5_dW14 = self.neuron5_dAct_dH5 * self.neuron2AH
        dH5_dW13 = self.neuron5_dAct_dH5 * self.neuron1AH
        dH5_dB5 = self.neuron5_dAct_dH5 * 1

        dH4_dW12 = self.neuron4_dAct_dH4 * self.neuron3AH
        dH4_dW11 = self.neuron4_dAct_dH4 * self.neuron2AH
        dH4_dW10 = self.neuron4_dAct_dH4 * self.neuron1AH
        dH4_dB4 = self.neuron4_dAct_dH4 * 1

        dH5_dH3 = self.neuron5_dAct_dH5 * self.weight15
        dH5_dH2 = self.neuron5_dAct_dH5 * self.weight14
        dH5_dH1 = self.neuron5_dAct_dH5 * self.weight13

        dH4_dH3 = self.neuron4_dAct_dH4 * self.weight12
        dH4_dH2 = self.neuron4_dAct_dH4 * self.weight11
        dH4_dH1 = self.neuron4_dAct_dH4 * self.weight10

        dH3_dW9 = self.neuron3_dAct_dH3 * self.input[2]
        dH3_dW8 = self.neuron3_dAct_dH3 * self.input[1]
        dH3_dW7 = self.neuron3_dAct_dH3 * self.input[0]
        dH3_dB3 = self.neuron3_dAct_dH3 * 1

        dH2_dW6 = self.neuron2_dAct_dH2 * self.input[2]
        dH2_dW5 = self.neuron2_dAct_dH2 * self.input[1]
        dH2_dW4 = self.neuron2_dAct_dH2 * self.input[0]
        dH2_dB2 = self.neuron2_dAct_dH2 * 1

        dH1_dW3 = self.neuron1_dAct_dH1 * self.input[2]
        dH1_dW2 = self.neuron1_dAct_dH1 * self.input[1]
        dH1_dW1 = self.neuron1_dAct_dH1 * self.input[0]
        dH1_dB1 = self.neuron1_dAct_dH1 * 1

        # update the weights
        self.weight17 = self.weight17 - self.learningRate * dLoss_dH6 * dH6_dW17
        self.weight16 = self.weight16 - self.learningRate * dLoss_dH6 * dH6_dW16
        self.weight15 = self.weight15 - self.learningRate * dLoss_dH6 * dH6_dH5 * dH5_dW15
        self.weight14 = self.weight14 - self.learningRate * dLoss_dH6 * dH6_dH5 * dH5_dW14
        self.weight13 = self.weight13 - self.learningRate * dLoss_dH6 * dH6_dH5 * dH5_dW13
        self.weight12 = self.weight12 - self.learningRate * dLoss_dH6 * dH6_dH4 * dH4_dW12
        self.weight11 = self.weight11 - self.learningRate * dLoss_dH6 * dH6_dH4 * dH4_dW11
        self.weight10 = self.weight10 - self.learningRate * dLoss_dH6 * dH6_dH4 * dH4_dW10
        self.weight9 = self.weight9 - self.learningRate * (dLoss_dH6 * dH6_dH5 * dH5_dH3 * dH3_dW9 + dLoss_dH6 * dH6_dH4 * dH4_dH3 * dH3_dW9)
        self.weight8 = self.weight8 - self.learningRate * (dLoss_dH6 * dH6_dH5 * dH5_dH3 * dH3_dW8 + dLoss_dH6 * dH6_dH4 * dH4_dH3 * dH3_dW8) 
        self.weight7 = self.weight7 - self.learningRate * (dLoss_dH6 * dH6_dH5 * dH5_dH3 * dH3_dW7 + dLoss_dH6 * dH6_dH4 * dH4_dH3 * dH3_dW7) 
        self.weight6 = self.weight6 - self.learningRate * (dLoss_dH6 * dH6_dH5 * dH5_dH2 * dH2_dW6 + dLoss_dH6 * dH6_dH4 * dH4_dH2 * dH2_dW6) 
        self.weight5 = self.weight5 - self.learningRate * (dLoss_dH6 * dH6_dH5 * dH5_dH2 * dH2_dW5 + dLoss_dH6 * dH6_dH4 * dH4_dH2 * dH2_dW5) 
        self.weight4 = self.weight4 - self.learningRate * (dLoss_dH6 * dH6_dH5 * dH5_dH2 * dH2_dW4 + dLoss_dH6 * dH6_dH4 * dH4_dH2 * dH2_dW4) 
        self.weight3 = self.weight3 - self.learningRate * (dLoss_dH6 * dH6_dH5 * dH5_dH1 * dH1_dW3 + dLoss_dH6 * dH6_dH4 * dH4_dH1 * dH1_dW3) 
        self.weight2 = self.weight2 - self.learningRate * (dLoss_dH6 * dH6_dH5 * dH5_dH1 * dH1_dW2 + dLoss_dH6 * dH6_dH4 * dH4_dH1 * dH1_dW2) 
        self.weight1 = self.weight1 - self.learningRate * (dLoss_dH6 * dH6_dH5 * dH5_dH1 * dH1_dW1 + dLoss_dH6 * dH6_dH4 * dH4_dH1 * dH1_dW1) 

        #update biases
        self.bias6 = self.bias6 - self.learningRate * dLoss_dH6 * dH6_dB6
        self.bias5 = self.bias5 - self.learningRate * dLoss_dH6 * dH6_dH5 * dH5_dB5
        self.bias4 = self.bias4 - self.learningRate * dLoss_dH6 * dH6_dH4 * dH4_dB4
        self.bias3 = self.bias3 - self.learningRate * (dLoss_dH6 * dH6_dH5 * dH5_dH3 * dH3_dB3 + dLoss_dH6 * dH6_dH4 * dH4_dH3 * dH3_dB3)
        self.bias2 = self.bias2 - self.learningRate * (dLoss_dH6 * dH6_dH5 * dH5_dH2 * dH2_dB2 + dLoss_dH6 * dH6_dH4 * dH4_dH2 * dH2_dB2)
        self.bias1 = self.bias1 - self.learningRate * (dLoss_dH6 * dH6_dH5 * dH5_dH1 * dH1_dB1 + dLoss_dH6 * dH6_dH4 * dH4_dH1 * dH1_dB1)


def testTwoSingle():
    testNN = hardcodedNN_twoIn_singleOut(0.08)
    trainingData = [[-2, -1], [25, 6], [17, 4], [17, 4], [-15, -6]]
    correctAnswers = [1, 0, 0, 1]

    for epoch in range(0, 1000):
        for data, correctAnswer in zip(trainingData, correctAnswers):
            result = testNN.forwardPropagate(data)
            testNN.backPropagate(correctAnswer, result)
    result = testNN.forwardPropagate([-2, -1])
    print("Prediction to [-2, -1]: %.3f" % result)
    result = testNN.forwardPropagate([25, 6])
    print("Prediction to [25, 6]: %.3f" % result)
    result = testNN.forwardPropagate([17, 4])
    print("Prediction to [17, 4]: %.3f" % result)
    result = testNN.forwardPropagate([-15, -6])
    print("Prediction to [-15, -6]: %.3f" % result)

def testThreeSingle():
    testNN = hardcodedNN_threeIn_singleOut(0.08)
    # training data: age, sex, fare
    trainingData = [[-2, -1, -10], [25, 6, 20], [17, 4, 15], [-15, -6, -5]]
    # gender based off height, weight, hairlength
    correctAnswers = [1, 0, 0, 1]
    for epoch in range(0, 1000):
        for data, correctAnswer in zip(trainingData, correctAnswers):
            result = testNN.forwardPropagate(data)
            testNN.backPropagate(correctAnswer, result)
    result = testNN.forwardPropagate([-2, -1, -10])
    print("\nPrediction to [-2, -1, -10]: %.3f" % result)
    result = testNN.forwardPropagate([25, 6, 20])
    print("Prediction to [25, 6, 20]: %.3f" % result)
    result = testNN.forwardPropagate([17, 4, 15])
    print("Prediction to [17, 4, 15]: %.3f" % result)
    result = testNN.forwardPropagate([-15, -6, -5])
    print("Prediction to [-15, -6, -5]: %.3f" % result)

    result = testNN.forwardPropagate([0,0,0])

testTwoSingle()
testThreeSingle()