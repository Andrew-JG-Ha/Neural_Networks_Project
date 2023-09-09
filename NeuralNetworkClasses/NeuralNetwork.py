from Layer import *

class NeuralNetwork():
    """
    Network of Layers
    """
    def __init__(self) -> None:
        self.network = []
        self.networkSize = 0

    def appendLayer(self, layerType:str, numberNeurons:int, numInputs:int = None, learningRate = 0.75, seed = None) -> None:
        numIns = numInputs
        if numIns == None and self.networkSize != 0:
            numIns = len(self.network[self.networkSize - 1].layer)
        newLayer = Layer(numIns, numberNeurons, layerType, learningRate, seed)
        self.network.append(newLayer)
        self.networkSize = self.networkSize + 1
        if (self.networkSize > 1):
            for i in range(0, self.networkSize - 1):
                self.network[i+1].setLayerInput([neuron.getNeuronOutput() for neuron in self.network[i].layer])

    def forwardPropagate(self, input:list):
        inputToLayer = input
        neuronOutput = []
        layerOutput = []
        for layerIndex in range(0, self.networkSize):
            for neuron in self.network[layerIndex].layer:
                for weight, neuronInput in zip(neuron.getWeights(), inputToLayer):
                    neuronOutput.append(weight*neuronInput)
                neuron.updateNeuronValue(sum(neuronOutput) + neuron.getBias())
                neuron.activate(self.network[layerIndex].activationType)
                neuron.activatePrime(self.network[layerIndex].activationType)
                neuronOutput = []
            layerOutput = [neuron.getNeuronOutput() for neuron in self.network[layerIndex].layer]
            self.network[layerIndex].setLayerInput(inputToLayer)
            if layerIndex < self.networkSize - 1:
                inputToLayer = layerOutput
                layerOutput = []
        return layerOutput

    """
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
    def backwardPropagate(self, expectedValues, actualValues) -> None:
        # Calculate the final layer's error
        msePrime = meanSquaredErrorPrime(expectedValues, actualValues) # dError/dPrediction = 2/n * (expected - actual)
        prev = []
        prevSize = 0
        dError_dW_Descending = []
        prevNeuronWeights = []
        currNeuronWeights = []
        for layerIndex in reversed(range(0, self.networkSize)):
            prevNeuronOutputs = [output for output in reversed(self.network[layerIndex].getLayerInput())]
            for currentLayerNeuron in reversed(self.network[layerIndex].layer):
                currNeuronWeights.append(currentLayerNeuron.getWeights())
            currNeuronWeights = [weight for neuronWeight in currNeuronWeights for weight in neuronWeight]
            if (layerIndex == self.networkSize - 1):
                for neuronInput in prevNeuronOutputs:
                    dError_dW_Descending.append(msePrime * neuronInput)
                prev.append(msePrime)
            else:
                prevValues = []
                for prevIndex in range(0, len(prev)):
                    prev = [prev[prevIndex]*weight for weight in prevNeuronWeights] # [prev[0] * W1, W2, W3 ..., prev[1] * W1, W2, W3, ...]
                    for prevValue in prev:
                        prevValues.append(prevValue)

                for prevIndex in range(0, len(prev)):
                    for inputIndex in range(0, len(self.network[layerIndex].getLayerInput())):
                        dError_dW_Descending.append(prev[prevIndex] * prevNeuronOutputs[inputIndex])
                
                """
                currPrevLength = len(prev)
                for prevIndex in range(prevSize, currPrevLength):
                    for weight in prevNeuronWeights:
                        prev.append(prev[prevIndex]*weight)
                prevSize = currPrevLength
                currPrevLength = len(prev)
                for prevIndex in range(prevSize, currPrevLength):
                    for inputIndex in range(0, len(self.network[layerIndex].getLayerInput())):
                        dError_dW_Descending.append(prev[prevIndex] * prevNeuronOutputs[inputIndex])
                prevSize = currPrevLength
                """
            prevNeuronWeights = currNeuronWeights
            currNeuronWeights = []
        
        # Updating the current weights
        # Getting all the neurons into 1 list
        allNeurons = [neuron for layer in self.network for neuron in layer.layer]
        

        # dPrediction/dWeight
        # (dError/DWeight) = (dError/dPrediction)*...*(dPrediction/dInput)*(dInput/dWeight)

        # newWeight = oldWeight - learningRate * (dError/dWeight_x)
        # NotImplementedError

def meanSquaredError(expectedValues:list, actualValues:list):
    return (1/len(expectedValues))*pow(sum([(actual-expected) for expected, actual in zip(expectedValues, actualValues)]), 2)

def meanSquaredErrorPrime(expectedValues:list, actualValues:list):
    return (2/len(expectedValues))*(sum([(actual-expected) for expected, actual in zip(expectedValues, actualValues)]))

def train(network:NeuralNetwork, input):
    NotImplementedError

def test1():
    test = NeuralNetwork()
    test.appendLayer(layerType = "ReLu", numberNeurons = 2, numInputs = 2, learningRate = 0.75, seed = 105)
    test.appendLayer(layerType = "Sigmoid", numberNeurons = 2, learningRate = 0.25)
    test.appendLayer(layerType = "Sigmoid", numberNeurons = 1, learningRate = 0.5)
    res = test.forwardPropagate([1,2])
    test.backwardPropagate([1,2], res)

test1()