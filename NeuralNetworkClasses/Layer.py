from Neuron import *
class Layer():
    def __init__(self, layerType:str, numberOfNeurons:int) -> None:
        self.layerInput = []
        self.layerType = layerType
        self.layer = []
        if self.layerType == "Input":
            self.layer = [Neuron() for neuron in range(0, numberOfNeurons)]
    
    def updateInputLayer(self, newInputs:list):
        self.layerInput = newInputs
        for neuron, newIn in zip(self.layer, newInputs):
            neuron.updateNeuronInput(newIn)
            neuron.updateNeuronOutput(newIn)

    def getLayerType(self):
        return self.layerType

    def getLayerInput(self):
        return self.layerInput

class HiddenLayer(Layer):
    """
    Layer of Neurons 
    """
    def __init__(self, layerType:str, numberOfNeurons:int, numberOfPriorNeurons:int, learningRate:float, seed = None) -> None:
        super().__init__(layerType, numberOfNeurons)
        self.learningRate = learningRate
        if self.layerType != "Input":
            self.layer = [HiddenNeuron(numberOfPriorNeurons, seed) for neuron in range(0, numberOfNeurons)]
        
    def updateLayerWeights(self, newWeights:list):
        for neuron, weights in zip(self.layer, newWeights):
            neuron.updateWeights([(oldWeight - self.learningRate*newWeight) for oldWeight, newWeight in zip(weights, newWeights)])
    
    def updateLayerBias(self, newBiases:list):
        for neuron, bias in zip(self.layer, newBiases):
            neuron.updateBias(neuron.getBias - self.learningRate*bias)

    def updateHiddenLayerInput(self, newInputs:list):
        self.layerInput = newInputs
        for neuron in self.layer:
            neuron.updateNeuronInput(newInputs)

# test that the Layer class is working as intended
def testLayer():
    test = Layer(4, "ReLu", 99)
    print(test.layer)
    print(test.layerOutput)
    print(test.activationType)
