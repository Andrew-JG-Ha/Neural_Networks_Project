from Neuron import *
class Layer():
    """
    Layer of Neurons 
    """
    def __init__(self, numberOfNeurons:int, activationType:str, learningRate:float,  seed = None) -> None:
        self.layer = [Neuron(numberOfNeurons, seed) for neuron in range(0, numberOfNeurons)]
        self.layerInput = []
        self.activationType = activationType
        self.learningRate = learningRate
        
    def updateLayerWeights(self, newWeights:list):
        for neuron, weights in zip(self.layer, newWeights):
            neuron.updateWeights([(oldWeight - self.learningRate*newWeight) for oldWeight, newWeight in zip(weights, newWeights)])
    
    def updateLayerBias(self, newBiases:list):
        for neuron, bias in zip(self.layer, newBiases):
            neuron.updateBias(neuron.getBias - self.learningRate*bias)

    def setLayerInput(self, newInputs:list):
        self.layerInput = newInputs
        for neuron, input in zip(self.layer, newInputs):
            neuron.updateNeuronInput(input)

    def getActivationType(self):
        return self.activationType


# test that the Layer class is working as intended
def testLayer():
    test = Layer(4, "ReLu", 99)
    print(test.layer)
    print(test.layerOutput)
    print(test.activationType)
