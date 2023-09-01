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
                self.network[i+1].setLayerInput(self.network[i].getLayerOutput())

    def forwardPropagate(self, input:list):
        inputToLayer = input
        neuronOutput = []
        layerOutput = []
        for layerIndex in range(0, self.networkSize):
            for neuron in self.network[layerIndex].layer:
                for weight, neuronInput in zip(neuron.getWeights(), inputToLayer):
                    neuronOutput.append(weight*neuronInput)
                neuron.updateNeuronValue(sum(neuronOutput) + neuron.getBias())
                neuron.activate()
                neuronOutput = []
            layerOutput = [neuron.getNeuronOutput() for neuron in self.network[layerIndex].layer]
            if layerIndex < self.networkSize - 1:
                inputToLayer = layerOutput
                layerOutput = []
        return layerOutput

    def backwardPropagate(self) -> None:
        NotImplementedError

def train(network:NeuralNetwork, input):
    NotImplementedError

def test1():
    test = NeuralNetwork()
    test.appendLayer("ReLu", 2, 0.75, 105)
    test.appendLayer("sigmoid", 2, 0.25)
    print(test.forwardPropagate([1,2]))

test1()