from Layer import *

class NeuralNetwork():
    """
    Network of Layers
    """
    def __init__(self) -> None:
        self.network = []
        self.networkSize = 0

    def appendLayer(self, layerType:str, numberNeurons:int, learningRate, seed = None) -> None:
        newLayer = Layer(numberNeurons, layerType, learningRate, seed)
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

    dError/dW6 = dError/dH5 * dH5/dH4 * dH4/dW6 = msePrime * d(W7H3 + W8H4 + b5)/dH4 * d(W5H1 + W6H2 + b4)/dW6 = msePrime * W8 * H2 = prev[0] * W8 * H2

    dError/dW5 = dError/dH5 * dH5/dH4 * dH4/dW5 = msePrime * d(W7H3 + W8H4 + b5)/dH4 * d(W5H1 + W6H2 + b4)/dW5 = msePrime * W8 * H1 = prev[0] * W8 * H1

    dError/dW4 = dError/dH5 * dH5/dH3 * dH3/dW4 = msePrime * d(W7H3 + W8H4 + b5)/dH3 * d(W3H1 + W4H2 + b3)/dW4 = msePrime * W7 * H2 = prev[0] * W7 * H2

    dError/dW3 = dError/dH5 * dH5/dH3 * dH3/dW3 = msePrime * d(W7H3 + W8H4 + b5)/dH3 * d(W3H1 + W4H2 + b3)/dW3 = msePrime * W7 * H1 = prev[0] * W7 * H1

    dError/dW2 = dError/dH5 * dH5/dH4 * dH4/dH2 * dH2/dW2 = msePrime * d(W7H3 + W8H4 + b5)/dH4 * d(W5H1 + W6H2 + b4)/dH2 * d(W2I2 + b2)/dW2 = msePrime * W8 * W6 * I2 = prev[1] * W6 * I2

    dError/dW1 = dError/dH5 * dH5/dH3 * dH3/dH1 * dH1/dW1 = msePrime * d(W7H3 + W8H4 + b5)/dH3 * d(W3H1 + W4H2 + b3)/dH1 * d(W1I1 + b1)/dW1 = msePrime * W7 * W3 * I1 = prev[2] * W3 * I1
    """
    def backwardPropagate(self, expectedValues, actualValues) -> None:
        # Calculate the final layer's error
        mse = meanSquaredError(expectedValues, actualValues)
        msePrime = meanSquaredErrorPrime(expectedValues, actualValues) # dError/dPrediction = 2/n * (expected - actual)
        prev = []
        prevSize = 0
        dError_dW_Descending = []
        for layerIndex in range(self.networkSize, 1, -1):
            prevNeuronOutputs = []
            currNeuronWeights = []
            for currentLayerNeuron in reversed(self.network[layerIndex].layer):
                currNeuronWeights.append(currentLayerNeuron.getWeights())
                prevNeuronOutputs.append(currentLayerNeuron.getNeuronOutput())
            if (layerIndex == self.networkSize):
                for input in prevNeuronOutputs:
                    dError_dW_Descending.append(msePrime * input)
                prev.append(msePrime)
            else:
                for prevIndex in range(prevSize, len(prev)):
                    for weight in currNeuronWeights:
                        for input in prevNeuronOutputs:
                            
                prevSize = len(prev)
            
        # dPrediction/dWeight
        # (dError/DWeight) = (dError/dPrediction)*...*(dPrediction/dInput)*(dInput/dWeight)

        # applying the activation prime function to the 
        # neuron.getNeuronValue()

        # newWeight = oldWeight - learningRate * (dError/dWeight_x)
        NotImplementedError

def meanSquaredError(expectedValues:list, actualValues:list):
    return (1/len(expectedValues))*(sum([(actual-expected)^2 for expected, actual in zip(expectedValues, actualValues)]))

def meanSquaredErrorPrime(expectedValues:list, actualValues:list):
    return (2/len(expectedValues))*(sum([(actual-expected) for expected, actual in zip(expectedValues, actualValues)]))

def train(network:NeuralNetwork, input):
    NotImplementedError

def test1():
    test = NeuralNetwork()
    test.appendLayer("ReLu", 2, 0.75, 105)
    test.appendLayer("Sigmoid", 2, 0.25)
    test.appendLayer("Sigmoid", 1, 0.5)
    print(test.forwardPropagate([1,2]))

test1()