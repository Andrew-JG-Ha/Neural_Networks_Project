from Neuron import *
class Layer():
    """
    Layer of neurons 
    """
    def __init__(self, numberOfNeurons:int, activationType:str, prevLayerNeurons:int, layerNumber:int) -> None:
        self.neurons = []
        self.layerOutputs = []
        self.layerValues = []
        self.inputs = []
        self.numberOfNeurons = numberOfNeurons
        self.activationType = activationType
        self.prevLayerNeurons = prevLayerNeurons
        self.layerNumber = layerNumber
        for i in range(0, numberOfNeurons):
            self.neurons.append(Neuron(prevLayerNeurons, i))
            self.layerOutputs.append(self.neurons[i].getNeuronOutput())
            self.layerValues.append(self.neurons[i].getNeuronValue())

    def forwardPropagate(self, inputs:list) -> None:
        for index, neuron in enumerate(self.neurons):
            value = np.dot(neuron.weights, inputs) + neuron.bias
            activatedValue = activationFunction(self.activationType, value)
            neuron.updateNeuronValue(value)
            neuron.updateNeuronOutput(activatedValue)
            self.layerOutputs[index] = neuron.getNeuronOutput()
            self.layerValues[index] = neuron.getNeuronValue()
        self.inputs = inputs

    # def outputBackwardPropagate(self, predictedValue:list, trueValue:list) -> list:
    #     outputGradient = np.subtract(predictedValue, trueValue) * activationFunctionDeriv(self.activationType, self.inputs)
    #     # outputGradient = - (np.divide(trueValue, predictedValue) - np.divide(1 - trueValue, 1 - predictedValue))
    #     dWeight = np.dot(outputGradient, self.inputs)
    #     dBias = np.sum(outputGradient, axis=0, keepdims=False)
    #     return outputGradient, dWeight, dBias

    # def hiddenBackwardPropagate(self, error, priorDelta) -> list:
    #     currentLayerWeights = [neuron.getWeights() for neuron in self.neurons]
    #     # hiddenGradient = np.dot(priorLayerWeights, priorDelta) * activationFunctionDeriv(self.activationType, self.inputs)
    #     hiddenGradient = np.dot(np.transpose(currentLayerWeights), priorDelta) * activationFunctionDeriv(self.activationType, self.inputs)
    #     dWeight = np.dot(hiddenGradient, self.inputs)
    #     dBias = np.sum(hiddenGradient, axis = 0, keepdims=False)
    #     return hiddenGradient, dWeight, dBias

    # def backwardPropagate(self, trueValue, predictedValue):
    #     error = lossPrime(trueValue, predictedValue)
    #     currentLayerWeights = [neuron.getWeights() for neuron in self.neurons]
    #     inputError = np.dot(error, currentLayerWeights) # error here
    #     weightsError = np.dot(np.transpose([error]), [self.inputs] )
    #     biasError = np.sum(inputError)
    #     return inputError, weightsError, biasError
    
    # def backwardPropagate(self, outputError, learningRate):
    #     activatedDerivError = activationFunctionDeriv(self.activationType, self.inputs) * outputError
    #     layerWeights = [neuron.getWeights() for neuron in self.neurons]

    #     inputError = np.dot(activatedDerivError, np.transpose(layerWeights))
    #     weightsError = np.dot(np.transpose(self.inputs), outputError)
    #     test = 1
    #     return inputError

    def getLayerOutput(self) -> list:
        return self.layerOutputs

    def getLayerValues(self) -> list:
        return self.layerValues

    def updateLayerParameters(self, newWeights:list[list], newBiases:list, learningRate = 0.1):
        for neuron in self.neurons:
            neuron.updateAllWeights(neuron.weights - learningRate*newWeights)
            neuron.updateBias(neuron.bias - learningRate*newBiases)

def activationFunction(activationType, neuronValue):
    if (activationType == "sigmoid"):
        result = sigmoid(neuronValue)
    elif (activationType == "tanh"):
        result = tanh(neuronValue)
    elif (activationType == "ReLU"):
        result = ReLU(neuronValue)
    elif (activationType == "leaky ReLU"):
        result = leakyReLU(neuronValue)
    return result

def activationFunctionDeriv(activationType, neuronValue):
    result = []
    toBeActivated = neuronValue
    if type(neuronValue) is not list:
        toBeActivated = [toBeActivated]

    if (activationType == "sigmoid"):
        for value in toBeActivated:
            result.append(sigmoidDeriv(value))
    elif (activationType == "ReLU"):
        for value in toBeActivated:
            result.append(ReLUDeriv(value))
    result = result
    return result

def sigmoid(x):
    return 1/(1+np.exp(-x))

def ReLU(x):
    return np.maximum(0, x)

def leakyReLU(x):
    return np.maximum(0.1*x, x)

def tanh(x):
    return math.tanh(x)

def sigmoidDeriv(x):
    return sigmoid(x) * (1 - sigmoid(x))

def ReLUDeriv(x):
    if x < 0:
        return 0
    else:
        return 1