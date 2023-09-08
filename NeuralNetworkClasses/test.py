input1 = [24129.4, 24072.8, 24536, 25506.1, 26959, 28025.1, 28685.8, 30105.6, 31341.1, 33188, 35022.4, 36088.7, 37021.6, 38626.9, 40505.8, 43361.5, 45611.9, 47330.4, 48333.1, 46549.3, 48032.7, 49878.3, 50574.1, 52921.8, 54623.7, 53576.8, 53783.4, 50901.9]

input2 = [0.12, 0.18, 0.33, 0.41, 0.29, 0.44, 0.41, 0.22, 0.24, 0.31, 0.45, 0.34, 0.47, 0.63, 0.4, 0.41, 0.54, 0.63, 0.62, 0.54, 0.63, 0.54, 0.68, 0.62, 0.64, 0.52, 0.64, 0.7, 0.58, 0.62, 0.44, 0.73, 0.86, 0.99, 0.89]

def mean(inputs:list):
    result = 0
    for input in inputs:
        result += input
    return result/len(inputs)

def standardDeviation(inputs:list):
    result = 0
    meanValue = mean(inputs)
    for input in inputs:
        result = (input - meanValue)**2/(len(inputs) - 1)
    result = result ** (1/2)
    return result

def variance(inputs:list):
    result = 0
    meanValue = mean(inputs)
    for input in inputs:
        result = (input - meanValue)**2/(len(inputs) - 1)
    return result

q2std = standardDeviation(input1)
q2mean = mean(input1)

q3std = standardDeviation(input2)
q3variance = variance(input2)
q3mean = mean(input2)

print("Q2 Standard Deviation: " + str(round(q2std, 4)) + ", Mean: " + str(round(q2mean, 4)))
print("Q3 Standard Deviation: " + str(round(q3std, 4)) + ", Mean: " + str(round(q3mean, 4)), ", Variation: " + str(round(q3variance, 6)))
print (len(input1))