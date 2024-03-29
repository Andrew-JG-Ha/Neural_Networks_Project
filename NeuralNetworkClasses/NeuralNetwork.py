from Layer import *

class NeuralNetwork():
    """
    Network of Layers
    """
    def __init__(self, numInputs:int, errorMethod = "mse") -> None:
        self.network = []
        self.network.append(Layer("Input", numInputs))
        self.errorMethod = errorMethod
        self.networkSize = 1

    def appendLayer(self, layerType:str, numberNeurons:int = None, learningRate = 0.08, seed = None) -> None:
        numIns = self.network[self.networkSize - 1].getNumLines()
        self.networkSize = self.networkSize + 1
        if (layerType != "Output"):
            newLayer = HiddenLayer(layerType, numberNeurons, numIns, learningRate, seed)
            self.network.append(newLayer)
        else:
            if numIns > 1:
                self.network.append(Layer("Softmax", numIns))
                self.network[self.networkSize - 1].updateInputLayer(self.network[self.networkSize - 2].getLayerOutput())
            else:
                self.network.append(Layer("Output", numIns))
                self.network[self.networkSize - 1].updateInputLayer(self.network[self.networkSize - 2].getLayerOutput())

    def forwardPropagate(self, nnInput:list):
        previousLayerOutput = nnInput
        for layer in self.network:
            layer.updateInputLayer(previousLayerOutput)
            previousLayerOutput = layer.getLayerOutput()
        return previousLayerOutput

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

    dError/db5 = dError/dH5 * dH5/db5 = msePrime * d(W7H3 + W8H4 + b5)/db5 = prev[0] * 1

    dError/db4 = dError/dH5 * dH5/dH4 * dH4/db4 = msePrime * d(W7H3 + W8H4 + b5)/dH4 * d(W5H1 + W6H2 + b4)/db4 = prev[0] * W8 = prev[1]

    dError/db3 = dError/dH5 * dH5/dH3 * dH3/db3 = msePrime * d(W7H3 + W8H4 + b5)/dH3 * d(W3H1 + W4H2 + b3)/db3 = prev[0] * W7 = prev[2]
    """
    def backPropagateWeights(self, expectedValues, actualValues) -> None:
        # outputLayer - perform mean squared error
        outputError = self.backPropagateOutputLayer(expectedValues, actualValues)

        # hiddenLayers - propagate the error until the first layer
        dError_dW_List, dError_db_List = self.backPropagateHiddenLayer(outputError)

        # update the weights
        self.updateWeights(dError_dW_List)
        self.updateBiases(dError_db_List)

    def backPropagateOutputLayer(self, expectedValues, actualValues):
        # Calculate the final layer's error
        error = None
        if self.errorMethod == "mse":
            error = meanSquaredErrorPrime(expectedValues, actualValues) # dError/dPrediction = 2/n * (expected - actual)
        elif self.errorMethod == "meanAbsoluteError":
            error = meanAbsoluteErrorPrime(expectedValues, actualValues)
        return error

    def backPropagateHiddenLayer(self, outputError):
        dError_dH_List = [[outputError]]
        dError_dH_Tracker = 0
        
        dError_dW_List = [] # will have an input, output and all the hidden layers
        for incrementIndex in range(1, self.networkSize - 1):
            layerIndex = (self.networkSize - 1) - incrementIndex

            dError_dW_Layer = []
            dError_dH_Layer = []

            layer = self.network[layerIndex]
            layerWeights = [neuron.getWeights() for neuron in layer.layer]
            layerPrimes = [neuron.getNeuronPrimeValue() for neuron in layer.layer]
            for errors, neuronWeights, activationDeriv in zip(dError_dH_List[dError_dH_Tracker], layerWeights, layerPrimes):
                dError_dW_Neuron = []
                for error in errors:
                    for layerInput in layer.layerInput:
                        dError_dW_Neuron.append(error * activationDeriv * layerInput)
                    dError_dW_Layer.append(dError_dW_Neuron)

                    dError_dH_Neuron = []
                    for weight in neuronWeights:
                        dError_dH_Neuron.append(error * activationDeriv * weight)
                    dError_dH_Layer.append(dError_dH_Neuron)

            dError_dH_List.append(dError_dH_Layer)
            dError_dW_List.append(dError_dW_Layer)  

            dError_dH_Tracker = dError_dH_Tracker + 1

        return dError_dW_List, dError_dH_List
            
    def updateWeights(self, dError_dW_List):
        for layerIndex in range(1, self.networkSize - 1):
            dError_List_Index = layerIndex - 1
            currentLayer = self.network[layerIndex]
            for neuron, dWeights in zip(currentLayer.layer, dError_dW_List[dError_List_Index]):
                currentWeights = neuron.getWeights()
                newWeights = [current - currentLayer.learningRate * new for current, new in zip(currentWeights, dWeights)]
                neuron.updateWeights(newWeights)

    def updateBiases(self, dError_dB_List):
        for layerIndex in range(1, self.networkSize - 1):
            errors = dError_dB_List[layerIndex]
            verticalSum = errors.pop()
            while errors:
                nextValues = errors.pop()
                for index in range(0, len(verticalSum)):
                    verticalSum[index] = verticalSum[index] + nextValues[index]
            currentLayer = self.network[layerIndex]
            for neuron, biasVal in zip(currentLayer.layer, verticalSum):
                neuron.updateBias(neuron.getBias() - currentLayer.learningRate * biasVal)
                
def meanSquaredError(expectedValues:list, actualValues:list):
    return (1/len(expectedValues))*pow(sum([(expected-actual) for expected, actual in zip(expectedValues, actualValues)]), 2)

def meanSquaredErrorPrime(expectedValues:list, actualValues:list):
    return ([-(2/len(expectedValues))*(expected-actual) for expected, actual in zip(expectedValues, actualValues)])

def meanAbsoluteError(expectedValues:list, actualValues:list):
    return (1/len(expectedValues)) * sum([abs(expected-actual) for expected, actual in zip(expectedValues, actualValues)])

def meanAbsoluteErrorPrime(expectedValues:list, actualValues:list):
    return [1 if actual > expected else -1 for expected, actual in zip(expectedValues, actualValues)]

def train(network:NeuralNetwork, traningData, correctAnswers):
    NotImplementedError

def test1():
    trainingData = [[-2, -1], [25, 6], [17, 4], [17, 4], [-15, -6]]
    correctAnswers = [[1], [0], [0], [1]]

    test = NeuralNetwork(2)
    test.appendLayer("Sigmoid", 3, 0.1)
    test.appendLayer("ReLu", 2, 0.25)
    test.appendLayer("Output")

    for epoch in range(0, 1000):
        result = test.forwardPropagate([1,0,0])
        test.backPropagateWeights([1, 0], result)

    # for epoch in range(0, 1000):
    #     for data, correctAnswer in zip(trainingData, correctAnswers):
    #         result = test.forwardPropagate(data)
    #         test.backPropagateWeights(correctAnswer, result)

    result = test.forwardPropagate([-2, -1])
    result = test.forwardPropagate([25, 6])
    result = test.forwardPropagate([17, 4])
    result = test.forwardPropagate([-15, -6])
    hello =0

test1()