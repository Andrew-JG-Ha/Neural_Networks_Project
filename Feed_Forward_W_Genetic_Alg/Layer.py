from Neuron import *

class Layer():
    """
    Layer of neurons 
    """
    def __init__(self, numberOfNeurons:int, activationType:str, prevLayerNeurons:int, layerNumber:int) -> None:
        self.neurons = []
        self.layerOutput = []
        self.numberOfNeurons = numberOfNeurons
        self.activationType = activationType
        self.prevLayerNeurons = prevLayerNeurons
        self.layerNumber = layerNumber
        for i in range(0, numberOfNeurons):
            self.neurons.append(Neuron(prevLayerNeurons, activationType, layerNumber, i))
            self.layerOutput.append(self.neurons[i].getNeuronOutput())
    
    def updateLayerValues(self, inputs:list) -> None:
        for index, neuron in enumerate(self.neurons):
            result = neuron.calculateNeuronOutput(inputs)
            self.layerOutput[index] = result

    def getLayerOutput(self) -> list:
        return self.layerOutput
    

test = Layer(3, "ReLU", 3, 0)
test.updateLayerValues([1,2,3])
test.updateLayerValues([20,30,40])
test.updateLayerValues([1,2,3])
print("hi")
