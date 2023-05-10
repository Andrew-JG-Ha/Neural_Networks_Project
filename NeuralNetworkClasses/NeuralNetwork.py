from Layer import *

class NeuralNetwork():
    def __init__(self, numberOfInputs:int, layerNeuronCounts:list[int], layerTypes:list[str]) -> None:
        self.numberOfInputs = numberOfInputs
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

    def forwardPropagate(self, inputs:list):
        if (self.numberOfInputs != len(inputs)):
            print("Inputs are not the same size as the required inputs")
        else:
            inputToLayer = inputs
            for layer in self.layers:
                layer.forwardPropagate(inputToLayer)
                inputToLayer = layer.getLayerOutput()
            self.output = self.layers[len(self.layers) - 1].getLayerOutput()

    def backwardPropagate(self, trueValue:list, learningRate):
        previousLayer = None
        previousGradient = None
        dWeight = None
        dBias = None
        inputError, weightsError, biasError = self.layers[len(self.layers) - 1].backwardPropagate(trueValue, self.output)
        for index, layer in enumerate(reversed(self.layers)):
            inputError, weightsError, biasError = layer.backwardPropagate(inputError)
            

        # for index, layer in enumerate(reversed(self.layers)): #need to go in reverse, do the nth layer before doing the nth-1 layer
        #     if index - 1 < 0:
        #         previousGradient, dWeight, dBias = layer.outputBackwardPropagate(self.output, trueValue)
        #     else:
        #         previousWeights = [neuron.weights for neuron in previousLayer.neurons]
        #         previousGradient, dWeight, dBias = layer.hiddenBackwardPropagate(previousWeights, previousGradient)
        #     previousLayer = layer
        #     layer.updateLayerParameters(dWeight, dBias, learningRate)

    def loss(self, expected:list):
        predictedValues = np.array(self.output)
        expectedValues = np.array(expected)
        return np.square(expectedValues - predictedValues).mean()

    def train(self, inputs:list, expectedOutput:list, epochs, learningRate=0.1):
        costHistory = []
        accuracyHistory = []
        
        for i in range(epochs):
            # forward propagate
            # get loss values and append to cost history
            # backwards propagate and update parameters
            for j in range(len(inputs)):
                self.forwardPropagate(inputs[j])
                # costHistory.append(self.loss(expectedOutput[j]))
                self.backwardPropagate(expectedOutput[j], learningRate)
        # print(costHistory)

    def predict(self, inputs):
        self.forwardPropagate(inputs)
        return self.output

x_train = [[0,0], [0,1], [1,0], [1,1]]
y_train = [[0], [1], [1], [0]]

testLayerNeuronCount = [2, 3, 1]
testLayerTypes = ["ReLU", "ReLU", "sigmoid"]

test = NeuralNetwork(2, testLayerNeuronCount, testLayerTypes)
test.train(x_train, y_train, 1000)
print(test.predict([0,0]))
