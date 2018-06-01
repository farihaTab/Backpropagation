import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
import numpy as np
import random
import sys, os

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

# dataset = [
#     [0, 0, 0],
#     [0, 1, 1],
#     [1, 0, 1],
#     [1, 1, 0]
# ]
# dataset = [
#         [2.7810836, 2.550537003, 0],
#         [1.465489372, 2.362125076, 0],
#         [3.396561688, 4.400293529, 0],
#         [1.38807019, 1.850220317, 0],
#         [3.06407232, 3.005305973, 0],
#         [7.627531214, 2.759262235, 1],
#         [5.332441248, 2.088626775, 1],
#         [6.922596716, 1.77106367, 1],
#         [8.675418651, -0.242068655, 1],
#         [7.673756466, 3.508563011, 1]
#         ]
seed = 100

class NeuralNetwork:
    def __init__(self, nHidden, hiddenLayerInfo, X, y, testX, testy, learnRate, alpha, nLoop):
        self.L = nHidden+2
        self.layerInfo = hiddenLayerInfo
        self.layerInfo.insert(0,len(X[0])) #nInputLayer
        self.layerInfo.append(len(pandas.unique(y)))
        self.learnRate = learnRate #learning rate
        self.alpha = alpha
        self.nLoop = nLoop
        self.N = len(X) #total samples
        self.X = X # features
        self.y = y # true output
        self.testX = testX
        self.testy = testy
        self.weights = None
        self.initialize()
        self.trainModelStochastic()
        enablePrint()
        # self.trainModel()
        self.testModel()

    def initialize(self):
        random.seed(seed)
        self.weights = [[] for i in range(self.L)]
        # print(self.weights)
        # print(self.weights[0]['layer'].append(2))
        # print(self.weights[0].get('layer'))
        for r in range(1,self.L):
            # print("r: " + repr(r) + 'th layer')
            currNeurons = self.layerInfo[r]
            prevNeurons = self.layerInfo[r-1]
            x = np.array([[random.random() for _ in range(prevNeurons+1)] for _ in range(currNeurons)]) #+1 for bias
            self.weights[r] = x
        for r in range(1,self.L):
            print("layer "+repr(r))
            for j in range( self.layerInfo[r]):
                for k in range(len(self.weights[r][j])):
                    t = '%.64f'%self.weights[r][j][k]
                    sys.stdout.write(t)
                    sys.stdout.write(' ')
                print()
            print()

    def forwardPropagate(self,row):
        outputs = [[] for _ in range(self.L)]
        outputs[0] = row
        for r in range(1,self.L):
            print("r: " + repr(r) + 'th layer')
            currNeurons = self.layerInfo[r]
            prevNeurons = self.layerInfo[r - 1]
            input = outputs[r - 1]
            input.append(1) #for bias
            for j in range(currNeurons):
                print("j: " + repr(j) + 'th neuron')
                weight = self.weights[r][j]
                output = self.calcOutput(input,weight)
                outputs[r].append(output)
                print(output)
            # print(outputs[r])
        return outputs

    def calcOutput(self,input,weight):
        net = np.multiply(input,weight)
        net = np.sum(net)
        output = self.sigmoid(self.alpha,net)
        return output

    def backwardPropagate(self,outputs,expected):
        # print('back')
        expectedOutputs = [0 for _ in range(self.layerInfo[-1])]
        expectedOutputs[int(expected)] = 1
        deltas = [[] for _ in range(self.L)]
        for r in range(self.L-1,0,-1):
            print("r: " + repr(r) + 'th layer')
            currNeurons = self.layerInfo[r]
            prevNeurons = self.layerInfo[r - 1]
            # input('enter')
            if r == self.L-1:
                error = np.subtract(expectedOutputs,outputs[r])
                t = np.subtract(1,outputs[r])
                t = np.multiply(outputs[r],t)
                sigmoidDiff = np.multiply(self.alpha,t)
                delta = np.multiply(error,sigmoidDiff)
                # print(outputs[r])
                # print(expectedOutputs)
                # print(error)
                # print(sigmoidDiff)
                # print(delta)
                deltas[r] = delta
            else:
                delta = []
                nextNeurons = self.layerInfo[r+1]
                for j in range(currNeurons):
                    d = self.calcDelta(nextNeurons,deltas[r+1],self.weights[r+1],outputs[r][j],j)
                    delta.append(d)
                deltas[r] = delta
            for j in range(currNeurons):
                print("j: " + repr(j) + 'th neuron')
                print(deltas[r][j])
        return deltas

    def calcDelta(self,nextNeurons,nextDeltas,weights,output,j):
        s = 0.0
        for k in range(nextNeurons):
            try:
                s = s+nextDeltas[k]*weights[k][j]
            except IndexError:
                print('s '+repr(s))
                print('k '+repr(k))
                print('j '+repr(j))
                print('nextDeltas '+repr(nextDeltas))
                print('weights '+repr(weights))

        d = s*output*(1-output)
        return d

    def calculateDelWeight(self,delWeights,deltas,outputs):
        for r in range(1,self.L):
            # print("r: " + repr(r) + 'th layer')
            currNeurons = self.layerInfo[r]
            prevNeurons = self.layerInfo[r - 1]
            for j in range(currNeurons):
                for k in range(prevNeurons):
                    delWeights[r][j][k] = delWeights[r][j][k] - self.learnRate*deltas[r][j]*outputs[r-1][k]

    def updateWeights(self,delWeights):
        for r in range(1,self.L):
            # print("r: " + repr(r) + 'th layer')
            currNeurons = self.layerInfo[r]
            prevNeurons = self.layerInfo[r - 1]
            for j in range(currNeurons):
                for k in range(prevNeurons):
                    self.weights[r][j][k] = self.weights[r][j][k] + delWeights[r][j][k]

    def updateWeightStochastic(self,deltas,outputs):
        for r in range(1,self.L):
            print("r: " + repr(r) + 'th layer')
            currNeurons = self.layerInfo[r]
            prevNeurons = self.layerInfo[r - 1]
            for j in range(currNeurons):
                print("j: " + repr(j) + 'th neuron');
                for k in range(prevNeurons):
                    self.weights[r][j][k] += self.learnRate*deltas[r][j]*outputs[r-1][k]
                    print( self.weights[r][j][k])
                # print(self.learnRate)
                # print("{:.64f}".format(self.weights[r][j][-1]))
                # print("{:.64f}".format(deltas[r][j]))
                # print("{:.64f}".format(self.learnRate*deltas[r][j]))
                self.weights[r][j][-1] += self.learnRate*deltas[r][j]
                print("{:.64f}".format(self.weights[r][j][-1]))

    def trainModel(self):
        for loop in range(self.nLoop):
            enablePrint()
            print('loop '+repr(loop))
            blockPrint()
            delWeights = [[] for i in range(self.L)]
            for r in range(1, self.L):
                # print("r: " + repr(r) + 'th layer')
                currNeurons = self.layerInfo[r]
                prevNeurons = self.layerInfo[r - 1]
                x = np.array([[0.0 for _ in range(prevNeurons + 1)] for _ in range(currNeurons)])  # +1 for bias
                delWeights[r] = x
            for i in range(self.N):
                row = self.X[i,:].tolist()
                outputs = self.forwardPropagate(row)
                deltas = self.backwardPropagate(outputs,self.y[i])
                self.calculateDelWeight(delWeights,deltas,outputs)
            self.updateWeights(delWeights)

    def trainModelStochastic(self):
        # print(self.weights)
        for loop in range(self.nLoop):
            enablePrint()
            print('loop '+repr(loop))
            blockPrint()
            for i in range(self.N):
                row = self.X[i,:].tolist()
                print('row: ' + repr(row))
                outputs = self.forwardPropagate(row)
                # print('outputs');print(outputs)
                # input('forward prop completed. enter ')
                deltas = self.backwardPropagate(outputs,self.y[i])
                # print('deltas');print(deltas)
                # input('backward prop completed. enter ')
                self.updateWeightStochastic(deltas,outputs)
                # input('updated weight. enter ')
                # print(self.weights)
                # input('enter')

    def predict(self,row):
        outputs = self.forwardPropagate(row)
        predicted = outputs[self.L-1]
        return predicted.index(max(predicted))

    def testModel(self):
        blockPrint()
        accuracy = 0.0
        for i in range(len(self.testX)):
            row = self.X[i, :].tolist()
            predict = self.predict(row)
            enablePrint()
            print(predict, self.testy[i])
            blockPrint()
            if predict == self.testy[i]:
                accuracy = accuracy + 1
        accuracy = accuracy/len(self.testX)*100
        enablePrint()
        print('accuracy '+repr(accuracy))

    def sigmoid(self, alpha, out):
        return 1.0 / (1 + np.exp(-alpha * out))

    def sigmoidDiff(self, alpha, out):
        return alpha*self.sigmoid(alpha, out) * (1 - self.sigmoid(alpha, out))

def normalize(dataset):
    array = np.array(dataset)
    max_min = [[max(array[:, i]), min(array[:, i])] for i in range(len(array[0]))]
    for i in range(len(array)):
        for j in range(len(array[0])-1): # output column wont be normalized
            dataset[i][j] = (array[i][j]-max_min[j][1])/(max_min[j][0]-max_min[j][1])

def takeInput():
    L = int(input("Enter number of hidden layers: "))
    hlayerInfo = []
    for i in range(L):
        msg = "no of neurons in %dth  hidden layer:" % (i+1)
        hlayerInfo.append(int(input(msg)))
    return L,hlayerInfo

def experiment3():
    url = "G:\Projects\PycharmProjects\PatternRecognition\BackPropagation\iris.data"
    dataset = pandas.read_csv(url)
    array = dataset.values
    np.random.seed(seed)
    np.random.shuffle(array)
    # normalize(array)
    X = array[:, :-1]
    Y = array[:, -1]
    Y = np.subtract(Y,1)
    validation_size = 0.20
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
    # nHidden = 1
    # hLayerInfo = [5]
    nHidden,hLayerInfo = takeInput()
    alpha = 1.0
    learningRate = 0.3
    nLoop = 100
    NeuralNetwork(nHidden, hLayerInfo, X_train, Y_train, X_validation, Y_validation, learningRate, alpha, nLoop)

def experiment1():
    url = "G:\Projects\PycharmProjects\PatternRecognition\BackPropagation\wheat-seeds.csv"
    dataset = pandas.read_csv(url)
    array = dataset.values
    normalize(array)
    X = array[:, :-1]
    Y = array[:, -1]
    Y = np.subtract(Y,1)
    validation_size = 0.20
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
    # nHidden = 1
    # hLayerInfo = [5]
    nHidden,hLayerInfo = takeInput()
    alpha = 1.0
    learningRate = 0.3
    nLoop = 10000
    NeuralNetwork(nHidden, hLayerInfo, X_train, Y_train, X_validation, Y_validation, learningRate, alpha, nLoop)

def experiment2():
    nHidden = 1
    hLayerInfo = [5]
    nHidden, hLayerInfo = takeInput()
    alpha = 1.0
    learningRate = 0.3
    nLoop = 100000
    dataset = [
        [0, 0, 0],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0]
    ]
    # dataset = [
    #     [2.7810836, 2.550537003, 0],
    #     [1.465489372, 2.362125076, 0],
    #     [3.396561688, 4.400293529, 0],
    #     [1.38807019, 1.850220317, 0],
    #     [3.06407232, 3.005305973, 0],
    #     [7.627531214, 2.759262235, 1],
    #     [5.332441248, 2.088626775, 1],
    #     [6.922596716, 1.77106367, 1],
    #     [8.675418651, -0.242068655, 1],
    #     [7.673756466, 3.508563011, 1]
    # ]
    # test = [
    #     [1.38807019, 1.850220317, 0],
    #     [3.06407232, 3.005305973, 0],
    #     [7.627531214, 2.759262235, 1],
    #     [5.332441248, 2.088626775, 1],
    # ]
    test = [
        [0, 0, 0],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0]
    ]
    np.random.seed(seed)
    np.random.shuffle(dataset)
    dataset = np.array(dataset)
    test = np.array(test)
    NeuralNetwork(nHidden, hLayerInfo, dataset[:, 0:2], dataset[:, 2], test[:, 0:2], test[:, 2], learningRate,
                  alpha, nLoop)


# experiment1()
experiment2()
# experiment3()


