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
            print('loop '+repr(loop))
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

    def configure(self):
        for r in range(self.L):
            print('r: '+repr(r))
            k_r = self.layerInfo[r + 1]  # num of neurons in layer r
            k_r_minus_1 = self.layerInfo[r]+1 # plus 1 for bias
            # for j in range(k_r):

            w = np.zeros(shape=[k_r,k_r_minus_1])
            for m in range(k_r):
                for n in range(k_r_minus_1):
                    w[m][n] = np.random.normal(0.0,0.1)
                # d = np.zeros(shape=[k_r,k_r_minus_1])
            self.weights.append(w)
            # self.deltas.append(d)
            self.deltas.append([0.0] * (self.layerInfo[r + 1]))
            self.nets.append([0.0]*(self.layerInfo[r+1]))
            self.outputs.append([0.0]*(self.layerInfo[r+1]))
            # self.deltas.append([0.0]*(self.layerInfo[r+1]))
            print("weights");print(self.weights)
        self.print()

        for i in range(self.N):
            print('i: '+repr(i)+'th sample')
            truey = self.y[i]
            # print(truey)
            for j in range(self.layerInfo[self.L]):
                if(truey==j):
                    self.trueY[i].append(1.0)
                else:
                    self.trueY[i].append(0.0)
        print('trueY');print(self.trueY)
        # self.test()
        return

    def forwardComputation(self,i,x):
        # for i in range(self.N): #ith sample
        x_i = x# self.X[i, :]
        print('x_i ');print(x_i);print(x)
        out = x_i #prev layer output
        k_r_minus_1 = self.layerInfo[0] # total input features
        # print(out.shape);
        for r in range(self.L): #rth layer
            print("\nr: " + repr(r+1) + 'th layer')
            out = np.append(out,[1.0])
            k_r = self.layerInfo[r+1] # num of neurons in layer r
            for j in range(k_r):
                print('j: '+repr(j)+'th neuron')
                weightV = self.weights[r][j,:]
                net = np.multiply(out,weightV)
                print('net layer '+repr(r+1)+' neuron '+repr(j));print(net);
                net = np.sum(net)
                print('net '+repr(net))
                self.nets[r][j] = net
                self.outputs[r][j] = self.sigmoid(self.alpha,net)
            out = self.outputs[r]
            print('out layer '+repr(r+1)+' neuron '+repr(j));print(out)
        self.print()

    def backwardComputation(self,i,outputs,nets):
        print('outputs');print(outputs)
        k_L = self.layerInfo[self.L]
        k_r_minus_1 = self.layerInfo[self.L-1]
        print("\nL: " + repr(self.L) + 'th layer')
        for j in range(k_L):
            print('\nj: ' + repr(j) + 'th neuron')
            print('output '+repr(outputs[self.L-1][j])+' true: '+repr(self.trueY[i][j]))
            e_j_i = outputs[self.L-1][j]-self.trueY[i][j]
            print('e_j_i: '+repr(e_j_i))
            f_prime_v_j_L = self.alpha*outputs[self.L-1][j]*(1.0-outputs[self.L-1][j])#self.sigmoidDiff(self.alpha,self.outputs[self.L-1][j])
            print('f_prime_v_j_L: '+repr(f_prime_v_j_L))
            # s = len(self.deltas[self.L - 1][j])
            self.deltas[self.L-1][j] = e_j_i*f_prime_v_j_L
            # self.deltas[self.L-1][j,:] = [f_prime_v_j_L]*s
            print('delta: '+ repr(self.deltas[self.L-1][j])+'for L: '+repr(self.L)+'th layer j: '+repr(j)+'th neuron')
        print('deltas:');print(self.deltas)

        for r in range(self.L-1,0,-1):
            print("\nr-1: " + repr(r) + 'th layer')
            k_r = self.layerInfo[r+1]
            k_r_minus_1 = self.layerInfo[r]
            print('k_r: '+repr(k_r)+' k_r-1: '+repr(k_r_minus_1))
            for j in range(k_r_minus_1):
                print('\nj: ' + repr(j) + 'th neuron')
                f_prime_v_j_L = self.alpha*outputs[r-1][j]*(1.0-outputs[r-1][j])#self.sigmoidDiff(self.alpha,self.outputs[r-1][j])
                print('f_prime_v_j_L: '+repr(f_prime_v_j_L))
                s = 0
                for k in range(k_r):
                    print('k: ' + repr(k) )
                    delta_k_r = self.deltas[r][k]
                    w_kj_r = self.weights[r][k][j]
                    print(self.deltas)
                    print('delta_k_r '+repr(delta_k_r)+' w_kj_r '+repr(w_kj_r))
                    s = s+ delta_k_r*w_kj_r
                    print('s '+repr(s))
                delta_j_r_minus_1 = s*f_prime_v_j_L
                self.deltas[r-1][j] = delta_j_r_minus_1
                print('delta_j_r_minus_1 '+repr(delta_j_r_minus_1)+'for r: '+repr(r)+'th layer j: '+repr(j)+'th neuron')

        print('deltas ');print(self.deltas)

    def train(self):
        Nets = []
        Outputs = []
        Deltas = []
        print('\n\nforward propagate')
        for i in range(self.N):
            print('\nforward propagate i: ' + repr(i) + 'th sample')
            self.forwardComputation(i,self.X[i,:])
            Nets.append(self.nets)
            Outputs.append(self.outputs)

        print("Outputs");print(Outputs)
        print("Nets");print(Nets)
        print('\n\nbackward propagate');
        for i in range(self.N):
            self.backwardComputation(i,Outputs[i],Nets[i])
            Deltas.append(self.deltas)
        print('Deltas ');print(Deltas)
        print('\n\ncalculate del weights');
        # print('Outputs: '); print(Outputs)

        delWeights = []
        for r in range(self.L):
            # input('enter')
            print("\n\nr: " + repr(r+1) + 'th layer')
            k_r = self.layerInfo[r + 1]  # num of neurons in layer r
            k_r_minus_1 = self.layerInfo[r]+1 # plus 1 for bias
            weightV = self.weights[r]
            print('weightV');print(weightV)
            w = np.zeros(shape=[k_r, k_r_minus_1])
            for j in range(k_r):
                print('\nj: ' + repr(j) + 'th neuron')
                sum = [0.0 for _  in range(k_r_minus_1)]#bias 1
                print('sum');print(sum)
                for i in range(self.N):
                    print('i: '+repr(i)+'th sample')
                    # m = np.multiply()
                    d_j_r_i = Deltas[i][r][j]
                    print('d_j_r_i: '+repr(d_j_r_i))
                    if(r==0):
                        y_r_minus_1 = self.X[i,:]
                    else:
                        y_r_minus_1 = Outputs[i][r-1]
                    print('y_r_minus_1: ' + repr(y_r_minus_1))
                    y_r_minus_1 = np.append(y_r_minus_1, [1.0])
                    print('y_r_minus_1: '+repr(y_r_minus_1))
                    m = np.multiply(d_j_r_i,y_r_minus_1)
                    print('m');print(m)
                    sum = np.add(sum,m)
                    print('sum');print(sum)
                    # input('\t\t\tenter')
                sum = np.multiply(-self.learnRate, sum)
                print('del w for j'+repr(j)+'th neuron in layer '+repr(r+1));print(sum)
                w[j] = sum
            print('weight for layer '+repr(r+1));print(w)
            delWeights.append(w)
        print('\nDel Weights');print(delWeights)

        print('\n\nupdate weights');
        for r in range(self.L):
            print('r: '+repr(r))
            k_r = self.layerInfo[r + 1]  # num of neurons in layer r
            k_r_minus_1 = self.layerInfo[r]+1 # plus 1 for bias
            weightV = self.weights[r]
            delWeightV = delWeights[r]
            print(weightV);print(delWeightV)
            print(np.add(weightV,delWeightV))
            self.weights[r] = np.add(weightV,delWeightV)

        print('\nUpdated weights ');print(self.weights)

def testModel3():
    url = "G:\Projects\PycharmProjects\PatternRecognition\BackPropagation\wheat-seeds.csv"
    dataset = pandas.read_csv(url)
    array = dataset.values
    X = array[:, 0:7]
    Y = array[:, 7]
    Y = np.subtract(Y,1)
    validation_size = 0.20
    # dataset = np.array(dataset)
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
    nHidden = 5
    hiddenLayerInfo = [2,2,2,2,2]
    alpha = 1.0
    learningRate = 0.3
    # print(X)

    # NeuralNetwork(nHidden, hiddenLayerInfo, X, Y, None, None, learnRate=learningRate,alpha=alpha, nLoop=100)
    NeuralNetwork(nHidden, hiddenLayerInfo, X_train, Y_train, X_validation, Y_validation, learnRate=learningRate,alpha=alpha, nLoop=100)


nHidden = 1
hLayerInfo = [5]
alpha = 1.0
learningRate = 0.3
nLoop = 1000
url = "G:\Projects\PycharmProjects\PatternRecognition\BackPropagation\wheat-seeds.csv"
dataset = pandas.read_csv(url)
dataset = dataset.values
array = np.array(dataset)
X = array[:, 0:7]
Y = array[:, 7]
Y = np.subtract(Y, 1)
validation_size = 0.20
# dataset = np.array(dataset)
# X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size,
#                                                                                 random_state=seed)

NeuralNetwork(nHidden, hLayerInfo, X, Y, X, Y, learningRate,1.0, nLoop)
# NeuralNetwork(nHidden, hLayerInfo, dataset[:,0:2], dataset[:,2], dataset[:,0:2], dataset[:,2], learningRate,1.0, nLoop)

