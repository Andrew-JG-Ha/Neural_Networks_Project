from Layer import *

class NeuralNetwork():
    """
    Network of Layers
    """
    def __init__(self, numInputs:int) -> None:
        self.network = []
        self.network.append(Layer("Input", numInputs))
        self.errorMethod = "meanAbsoluteError"
        self.networkSize = 1

    def appendLayer(self, layerType:str, numberNeurons:int = None, learningRate = 0.08, seed = None) -> None:
        numIns = self.network[self.networkSize - 1].getNumLines()
        if (layerType != "Output"):
            newLayer = HiddenLayer(layerType, numberNeurons, numIns, learningRate, seed)
            self.network.append(newLayer)
            self.networkSize = self.networkSize + 1
            if (self.networkSize > 1):
                for i in range(0, self.networkSize - 1):
                    self.network[i+1].updateInputLayer(self.network[i].getLayerOutput())
        else:
            self.network.append(Layer("Output", numIns))
            self.networkSize = self.networkSize + 1
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
        if self.errorMethod == "msePrime":
            error = meanSquaredErrorPrime(expectedValues, actualValues) # dError/dPrediction = 2/n * (expected - actual)
        elif self.errorMethod == "meanAbsoluteError":
            error = meanAbsoluteError(expectedValues, actualValues)
        elif self.errorMethod == "rmsPrime":
            error = rootMeanSquareError(expectedValues, actualValues)
        errors = [expected - actual for expected, actual in zip(expectedValues, actualValues)]
        outputLayerError = [error * difference for difference in errors]
        return outputLayerError

    def backPropagateHiddenLayer(self, outputError):
        dError_dH_List = [[outputError]]
        dError_dH_Tracker = 0

        # allWeights = []
        # for layer in self.network:
        #     layerWeights = []
        #     for neuron in layer.layer:
        #         layerWeights.append(neuron.getWeights())
        #     allWeights.append(layerWeights)
        
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
        

    # def backPropagateHiddenLayer(self, outputError):
    #     dError_dW_List = [[]]
    #     dError_dH_List = [outputError]
    #     for currentLayerIndex in reversed(range(1, len(self.network) - 1)):
    #         # Calculate a layer's dError/dWeight
    #         # calculate the error at each layer
    #         numCurrLayerNeurons = len(self.network[currentLayerIndex].layer)
    #         # numPrevLayerNeurons = len(self.network[currentLayerIndex + 1].layer)
    #         prevLayerError = dError_dH_List[-1]
    #         prevLayerWeights = [neuron.getWeights() for neuron in self.network[currentLayerIndex + 1].layer]
    #         currLayerActivationPrime = [neuron.getActivatedOutput() for neuron in self.network[currentLayerIndex].layer]

    #         # dError/dLayer[1+1] = ((w[l] * dError/dLayer[l]) dot actPrime[l+1])
    #         allWeights = []
    #         for neuronWeights in prevLayerWeights:
    #             allWeights = allWeights + neuronWeights

    #         # transposing the weights
    #         transposedWeights = []
    #         for weightIndex in range(0, numCurrLayerNeurons):
    #             currNeuronWeights = []
    #             for index in range(weightIndex, len(allWeights), numCurrLayerNeurons):
    #                 currNeuronWeights.append(allWeights[index])
    #             transposedWeights.append(currNeuronWeights)

    #         # Calculate the error propagate
    #         error_propagate = []
    #         for weightIndex in range(0, len(transposedWeights)):
    #             error_propagate.append(sum([weight*error for weight, error in zip(transposedWeights[weightIndex], prevLayerError)]))
    #         dError_dH_List.append(error_propagate)

    #         dError_dW = []
    #         for actPrime in currLayerActivationPrime:
    #             for prevError in prevLayerError:
    #                 dError_dW.append(actPrime * prevError)
    #         dError_dW_List.append(dError_dW)

    #     return dError_dW_List, dError_dH_List
            
    def updateWeights(self, dError_dW_List):
        for layerIndex in range(1, self.networkSize - 1):
            dError_List_Index = layerIndex - 1
            currentLayer = self.network[layerIndex]
            for neuron, dWeights in zip(currentLayer.layer, dError_dW_List[dError_List_Index]):
                currentWeights = neuron.getWeights()
                newWeights = [current - currentLayer.learningRate * new for current, new in zip(currentWeights, dWeights)]
                neuron.updateWeights(newWeights)
    
    # def updateWeights(self, dError_dW_List):
    #     for layerIndex in reversed(range(1, len(self.network) - 1)):
    #         weightsLayer = dError_dW_List[layerIndex]
    #         if weightsLayer:
    #             currLayer = self.network[layerIndex + 1]
    #             for neuron in reversed(currLayer.layer):
    #                 newWeights = []
    #                 for oldWeight in reversed(neuron.getWeights()):
    #                     newWeights.append(oldWeight + currLayer.learningRate * weightsLayer.pop())
    #                 newWeights.reverse()
    #                 neuron.updateWeights(newWeights)

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

    # def updateBiases(self, dError_dB_List):
    #     for layerIndex in reversed(range(1, len(self.network) - 1)):
    #         biasesLayer = dError_dB_List[(len(dError_dB_List) - 1) - layerIndex]
    #         currLayer = self.network[layerIndex + 1]
    #         for neuron in reversed(currLayer.layer):
    #             oldBias = neuron.getBias()
    #             neuron.updateBias(oldBias + currLayer.learningRate * biasesLayer.pop(0))
                
def meanSquaredError(expectedValues:list, actualValues:list):
    return (1/len(expectedValues))*pow(sum([(actual-expected) for expected, actual in zip(expectedValues, actualValues)]), 2)

def meanSquaredErrorPrime(expectedValues:list, actualValues:list):
    return -(2/len(expectedValues))*(sum([(actual-expected) for expected, actual in zip(expectedValues, actualValues)]))

def meanAbsoluteError(expectedValues:list, actualValues:list):
    return (1/len(expectedValues)) * sum([abs(actual-expected) for expected, actual in zip(expectedValues, actualValues)])

def rootMeanSquareError(expectedValues:list, actualValues:list):
    return pow(meanSquaredError(expectedValues, actualValues), 1/2)

def train(network:NeuralNetwork, traningData):
    NotImplementedError

def test1():
    test = NeuralNetwork(2)
    test.appendLayer("ReLu", 3)
    test.appendLayer("ReLu", 3, 0.25)
    test.appendLayer("Output")
    print("Hello World!")

    for i in range(0, 10000):
        result = test.forwardPropagate([0,0,0])
        test.backPropagateWeights([0,1,0], result)
        
    result = test.forwardPropagate([0,0,0])
    result = test.forwardPropagate([1,1,1])


test1()