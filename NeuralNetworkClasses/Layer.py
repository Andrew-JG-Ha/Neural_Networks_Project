from Neuron import *
import math

class Layer():
    def __init__(self, layerType:str, numParameters:int) -> None:
        self.layerInput = []
        self.numParameters = numParameters
        self.layerType = layerType
        self.layerOutput = []
        self.layer = []

    def getLayerType(self):
        return self.layerType

    def getLayerInput(self):
        return self.layerInput
    
    def getLayerOutput(self):
        return self.layerOutput
    
    def getNumLines(self):
        return self.numParameters

    def updateInputLayer(self, newInputs:list):
        self.layerInput = newInputs
        if (self.layerType == "Input" or self.layerType == "Output"):
            self.layerOutput = newInputs
        else:
            # softmax
            # RESULT IS TOO LARGE ISSUE!?!?!?!?!
            exponentialInput = [pow(math.e, input) for input in self.layerInput]
            self.layerOutput = [input / sum(exponentialInput) for input in exponentialInput]

            # self.layerOutput = [input/ sum(self.layerInput) for input in self.layerInput]

class HiddenLayer(Layer):
    """
    Layer of Neurons 
    """
    def __init__(self, layerType:str, numberOfNeurons:int, numberOfPriorNeurons:int, learningRate:float = 0.08, seed = None) -> None:
        super().__init__(layerType, numberOfNeurons)
        self.learningRate = learningRate
        if self.layerType != "Input":
            self.layer = [Neuron(numberOfPriorNeurons, layerType, seed) for neuron in range(0, numberOfNeurons)]
        
    # def updateLayerWeights(self, newWeights:list):
    #     for neuron, weights in zip(self.layer, newWeights):
    #         neuron.updateWeights([(oldWeight - self.learningRate*newWeight) for oldWeight, newWeight in zip(weights, newWeights)])
    
    # def updateLayerBias(self, newBiases:list):
    #     for neuron, bias in zip(self.layer, newBiases):
    #         neuron.updateBias(neuron.getBias - self.learningRate*bias)

    def updateInputLayer(self, newInputs:list):
        if (self.layerType == "ReLu" or self.layerType == "Sigmoid" or self.layerType == "Tanh" or self.layerType == "Step"):
            self.layerInput = newInputs
            self.layerOutput = []
            for neuron in self.layer:
                neuron.updateNeuronInput(self.layerInput)
                self.layerOutput.append(neuron.getActivatedOutput())
        else:
            super().updateInputLayer(newInputs)

# test that the Layer class is working as intended
def testLayer():
    input = Layer("input")
    output = Layer("output")
    hidden = HiddenLayer("ReLu", 4, 2, 0.25)
    print("Before updating inputs: ")
    print(hidden.getLayerInput())
    print(hidden.getLayerOutput())

    hidden.updateInputLayer([1,-2])
    print("After updating inputs: ")
    print(hidden.getLayerInput())
    print(hidden.getLayerOutput())

# testLayer()
