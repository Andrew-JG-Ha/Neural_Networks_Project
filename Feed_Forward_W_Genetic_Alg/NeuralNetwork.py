from Layer import *

testLayerNeuronCount = [2,4,4,2,1]
testLayerTypes = ["ReLU", "ReLU", "ReLU", "ReLU", "sigmoid"]


class NeuralNetwork():
    def __init__(self, numberOfInputs:int, layerNeuronCounts:list[int], layerTypes:list[int]) -> None:
        self.layers = []
        if (len(layerNeuronCounts) != len(layerTypes)):
            print("Length of arrays: 'layerNeuronCounts' and 'layerTypes' are not the same")
        else:
            for index, package in enumerate(zip(layerNeuronCounts, layerTypes)):
                newLayer = None
                if (index - 1 < 0):
                    newLayer = Layer(package[0], package[1], numberOfInputs, index)
                else:
                    newLayer = Layer(package[0], package[1], layerNeuronCounts[index - 1], index)
                self.layers.append(newLayer)
        self.output = self.layers[len(layerNeuronCounts) - 1].getLayerOutput()

    def singleLayerForwardPropagation(self):
        pass

    def singleLayerBackwardPropagation(self):
        pass

test = NeuralNetwork(2, testLayerNeuronCount, testLayerTypes)
print("hi")