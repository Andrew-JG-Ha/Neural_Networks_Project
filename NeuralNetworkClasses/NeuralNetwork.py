"""
http://neuralnetworksanddeeplearning.com/chap2.html
https://towardsdatascience.com/math-neural-network-from-scratch-in-python-d6da9f29ce65
https://www.youtube.com/watch?v=0e0z28wAWfg
https://builtin.com/machine-learning/backpropagation-neural-network
https://www.youtube.com/watch?v=9RN2Wr8xvro
"""

from Layer import *

class NeuralNetwork():
    """
    Network of Layers
    """
    def __init__(self, numInputs:int) -> None:
        self.network = []
        self.network.append(Layer("Input", numInputs))
        self.networkSize = 1
        self.numNeurons = numInputs

    def appendLayer(self, layerType:str, numberNeurons:int, learningRate = 0.05, seed = None) -> None:
        numIns = len(self.network[self.networkSize - 1].layer)
        newLayer = HiddenLayer(layerType, numberNeurons, numIns, learningRate, seed)
        self.network.append(newLayer)
        self.networkSize = self.networkSize + 1
        self.numNeurons = self.numNeurons + numberNeurons
        if (self.networkSize > 1):
            for i in range(0, self.networkSize - 1):
                self.network[i+1].updateHiddenLayerInput([neuron.getNeuronOutput() for neuron in self.network[i].layer])

    def forwardPropagate(self, nnInput:list):
        inputToLayer = nnInput
        neuronOutput = []
        layerOutput = []
        # first layer (input layer)
        self.network[0].updateInputLayer(nnInput)
        # hidden layers
        for layerIndex in range(1, self.networkSize):
            inputToLayer = [neuron.getNeuronOutput() for neuron in self.network[layerIndex - 1].layer]
            self.network[layerIndex].updateHiddenLayerInput(inputToLayer)
            for neuron in self.network[layerIndex].layer:
                for weight, neuronInput in zip(neuron.getWeights(), inputToLayer):
                    neuronOutput.append(weight*neuronInput)
                neuron.updateNeuronValue(sum(neuronOutput) + neuron.getBias())
                neuron.activate(self.network[layerIndex].getLayerType())
                neuron.activatePrime(self.network[layerIndex].getLayerType())
                neuronOutput = []
            layerOutput = [neuron.getNeuronOutput() for neuron in self.network[layerIndex].layer]
            if layerIndex < self.networkSize - 1:
                inputToLayer = layerOutput
                layerOutput = []
        return layerOutput

    """
    DY/DX = DY/DP * DP/DX

    dError/dW8 = dError/dH5 * dH5/dW8 = msePrime * d(W7H3 + W8H4 + b5)/dW8 = prev[0] * H4

    dError/dW7 = dError/dH5 * dH5/dW7 = msePrime * d(W7H3 + W8H4 + b5)/dW7 = prev[0] * H3

    dError/dW6 = dError/dH5 * dH5/dH4 * dH4/dW6 = msePrime * d(W7H3 + W8H4 + b5)/dH4 * d(W5H1 + W6H2 + b4)/dW6 = prev[0] * W8 * H2 = prev[1] * H2

    dError/dW5 = dError/dH5 * dH5/dH4 * dH4/dW5 = msePrime * d(W7H3 + W8H4 + b5)/dH4 * d(W5H1 + W6H2 + b4)/dW5 = prev[0] * W8 * H1 = prev[1] * H1

    dError/dW4 = dError/dH5 * dH5/dH3 * dH3/dW4 = msePrime * d(W7H3 + W8H4 + b5)/dH3 * d(W3H1 + W4H2 + b3)/dW4 = prev[0] * W7 * H2 = prev[2] * H2

    dError/dW3 = dError/dH5 * dH5/dH3 * dH3/dW3 = msePrime * d(W7H3 + W8H4 + b5)/dH3 * d(W3H1 + W4H2 + b3)/dW3 = prev[0] * W7 * H1 = prev[2] * H1

    dError/dW2 = dError/dH5 * dH5/dH4 * dH4/dH2 * dH2/dW2 = msePrime * d(W7H3 + W8H4 + b5)/dH4 * d(W5H1 + W6H2 + b4)/dH2 * d(W2I2 + b2)/dW2 
    = prev[0] * W8 * W6 * I2 = prev[1] * W6 * I2 

    dError/dW1 = dError/dH5 * dH5/dH3 * dH3/dH1 * dH1/dW1 = msePrime * d(W7H3 + W8H4 + b5)/dH3 * d(W3H1 + W4H2 + b3)/dH1 * d(W1I1 + b1)/dW1 
    = prev[0] * W7 * W3 * I1 = prev[2] * W3 * I1
    """
    def backPropagateWeights(self, expectedValues, actualValues) -> None:
        dError_dW_List = []
        errorPropagate = []
        # outputLayer - perform mean squared error
        errorPropagate.append(self.backPropagateOutputLayer(expectedValues, actualValues, dError_dW_List))
        # hiddenLayers - propagate the error until the first layer
        self.backPropagateHiddenLayer(len(self.network) - 2, errorPropagate[0], dError_dW_List)
        # inputLayer

        # update the weights
        NotImplementedError

    def backPropagateOutputLayer(self, expectedValues, actualValues, dError_dW_List:list):
        # Calculate the final layer's error
        msePrime = meanSquaredErrorPrime(expectedValues, actualValues) # dError/dPrediction = 2/n * (expected - actual)
        errors = [expected - actual for expected, actual in zip(expectedValues, actualValues)]
        outputLayerError = [msePrime * error for error in errors]
        outputLayerWeights = [neuron.getWeights() for neuron in self.network[-1].layer]
        return outputLayerError

    def backPropagateHiddenLayer(self, currentLayerIndex:int, previousLayerError:list, dError_dW_List:list):
        # Calculate a layer's dError/dWeight
        if (self.network[currentLayerIndex].layerType == "Input"):
            return self.backPropagateInputLayer()
        else:
            # calculate the error at each layer
            numCurrLayerNeurons = len(self.network[currentLayerIndex].layer)
            # numPrevLayerNeurons = len(self.network[currentLayerIndex + 1].layer)
            prevLayerError = previousLayerError
            prevLayerWeights = [neuron.getWeights() for neuron in self.network[currentLayerIndex + 1].layer]
            currLayerActivationPrime = [neuron.getPrimeValue() for neuron in self.network[currentLayerIndex].layer]
            # dError/dLayer[1+1] = ((w[l] * dError/dLayer[l]) dot actPrime[l+1])
            allWeights = []
            for neuronWeights in prevLayerWeights:
                allWeights = allWeights + neuronWeights

            # transposing the weights
            transposedWeights = []
            for weightIndex in range(0, numCurrLayerNeurons):
                currNeuronWeights = []
                for index in range(weightIndex, len(allWeights), numCurrLayerNeurons):
                    currNeuronWeights.append(allWeights[index])
                transposedWeights.append(currNeuronWeights)
            weight_dError_list = []
            for transposedWeight in transposedWeights:
                weight_dError = []
                for weight, error in zip(transposedWeight, prevLayerError):
                    weight_dError.append(weight*error)
                weight_dError_list.append(sum(weight_dError))
            dError_dW = []
            for actPrime, weight_dError in zip(currLayerActivationPrime, weight_dError_list):
                dError_dW.append(actPrime * weight_dError)
            dError_dW_List.append(dError_dW)
            self.backPropagateHiddenLayer(currentLayerIndex - 1, dError_dW_List[-1], dError_dW_List)
            

    def backPropagateInputLayer(self, previousLayerError:list, dError_dW_List:list):
        NotImplementedError

    # def backPropagateWeights(self, expectedValues, actualValues) -> None:
    #     # Calculate the final layer's error
    #     msePrime = meanSquaredErrorPrime(expectedValues, actualValues) # dError/dPrediction = 2/n * (expected - actual)
    #     prev = []
    #     dError_dW_Descending = []
    #     prevNeuronWeights = []
    #     currNeuronWeights = []
    #     # computing the derivative of the error with respect to each weight
    #     for layerIndex in reversed(range(0, self.networkSize)):
    #         prevNeuronOutputs = [output for output in reversed(self.network[layerIndex].getLayerInput())]
    #         for currentLayerNeuron in reversed(self.network[layerIndex].layer):
    #             currNeuronWeights.append(currentLayerNeuron.getWeights())
    #         currNeuronWeights = [weight for neuronWeight in currNeuronWeights for weight in neuronWeight]
    #         if (layerIndex == self.networkSize - 1):
    #             errors = [actual - expected for expected, actual in zip(expectedValues, actualValues)]
    #             for error, neuronInput in zip(errors, prevNeuronOutputs):
    #                 dError_dW_Descending.append(error * msePrime * neuronInput)
    #             prev.append(msePrime)
    #         else:
    #             prevValues = []
    #             numNeurons = len(self.network[layerIndex].layer)
    #             for prevIndex in range(0, len(prev)):
    #                 prev = [prev[prevIndex]*weight for weight in prevNeuronWeights] # [prev[0] * W1, W2, W3 ..., prev[1] * W1, W2, W3, ...]
    #                 for prevValue in prev:
    #                     prevValues.append(prevValue)
    #             if len(prevValues) == numNeurons:
    #                 for prevIndex in range(0, len(prevValues)):
    #                     for inputIndex in range(0, len(self.network[layerIndex].getLayerInput())):
    #                         dError_dW_Descending.append(prevValues[prevIndex] * prevNeuronOutputs[inputIndex])
    #             else:
    #                 numWeights = len(self.network[layerIndex].layerInput) * numNeurons # provides the number of weights
    #                 partitionedList = [prevValues[i:i + numWeights] for i in range(0, len(prevValues), numWeights)]
    #                 prevValues = [0] * numWeights
    #                 for partition in partitionedList:
    #                     for partitionIndex in range(0, len(partition)):
    #                         prevValues[partitionIndex] = prevValues[partitionIndex] + partition[partitionIndex]
    #                 partitionedList = [prevValues[i:i + numNeurons] for i in range(0, len(prevValues), numNeurons)]
    #                 for partition in partitionedList:
    #                     for prevIndex, inputIndex in zip(range(0, len(partition)), range(0, len(self.network[layerIndex].getLayerInput()))):
    #                         dError_dW_Descending.append(partition[prevIndex] * prevNeuronOutputs[inputIndex])
    #         prevNeuronWeights = currNeuronWeights
    #         currNeuronWeights = []
    #     # updating the weights in each neuron
    #     # newWeight = oldWeight - learningRate * (dError/dWeight_x)
    #     for layer in reversed(self.network):
    #         newWeights = []
    #         for neuron in reversed(layer.layer):
    #             oldWeights = neuron.getWeights()
    #             for oldWeight in oldWeights:
    #                 newWeights.append(oldWeight - layer.learningRate * dError_dW_Descending.pop(0))
    #             neuron.updateWeights(newWeights)
    #             newWeights = []

def meanSquaredError(expectedValues:list, actualValues:list):
    return (1/len(expectedValues))*pow(sum([(actual-expected) for expected, actual in zip(expectedValues, actualValues)]), 2)

def meanSquaredErrorPrime(expectedValues:list, actualValues:list):
    return -(2/len(expectedValues))*(sum([(actual-expected) for expected, actual in zip(expectedValues, actualValues)]))

def train(network:NeuralNetwork, input):
    NotImplementedError

def test1():
    test = NeuralNetwork(2)
    test.appendLayer(layerType = "ReLu", numberNeurons = 2, seed = 105)
    test.appendLayer(layerType = "ReLu", numberNeurons = 2)
    test.appendLayer(layerType = "Sigmoid", numberNeurons = 1)

    for i in range(0, 100):
        res = test.forwardPropagate([1,2])
        test.backPropagateWeights([5], res)
    res = test.forwardPropagate([1,2])


test1()