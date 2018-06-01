import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
import numpy as np
import random
import sys, os
from csv import reader

# Load a CSV file
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

# Convert string column to float
def str_column_to_float(dataset, column):
    i = 1
    for row in dataset:
        # print(repr(i)+": "+repr(row));i=i+1
        row[column] = float(row[column].strip())

# Convert string column to integer
def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup

# Find the min and max values for each column
def dataset_minmax(dataset):
    minmax = list()
    stats = [[min(column), max(column)] for column in zip(*dataset)]
    return stats

# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
    for row in dataset:
        # print('row '+repr(row))
        for i in range(len(row)-1):
            # print('\ti: '+repr(i))
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

dataset = [[2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]]

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
        # print('weights '+repr(self.weights))

    def forwardPropagate(self,row):
        outputs = [[] for _ in range(self.L)]
        outputs[0] = row
        for r in range(1,self.L):
            # print("r: " + repr(r) + 'th layer')
            currNeurons = self.layerInfo[r]
            prevNeurons = self.layerInfo[r - 1]
            input = outputs[r - 1]
            input.append(1) #for bias
            for j in range(currNeurons):
                weight = self.weights[r][j]
                output = self.calcOutput(input,weight)
                outputs[r].append(output)
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
            # print("r: " + repr(r) + 'th layer')
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
                # print(delta)
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
            # print("r: " + repr(r) + 'th layer')
            currNeurons = self.layerInfo[r]
            prevNeurons = self.layerInfo[r - 1]
            for j in range(currNeurons):
                for k in range(prevNeurons):
                    self.weights[r][j][k] = self.weights[r][j][k] - self.learnRate*deltas[r][j]*outputs[r-1][k]


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
            # input('enter')
            print('loop '+repr(loop))
            for i in range(self.N):
                row = self.X[i,:].tolist()
                # print(row)
                outputs = self.forwardPropagate(row)
                # print('outputs');print(outputs)
                deltas = self.backwardPropagate(outputs,self.y[i])
                # print('deltas');print(deltas)
                self.updateWeightStochastic(deltas,outputs)
                # print(self.weights)
                # input('enter')


    def predict(self,row):
        outputs = self.forwardPropagate(row)
        predicted = outputs[self.L-1]
        return predicted.index(max(predicted))

    def testModel(self):
        accuracy = 0.0
        for i in range(len(self.testX)):
            row = self.X[i, :].tolist()
            predict = self.predict(row)
            print('predicted ' + repr(predict) + ' true ' + repr(self.testy[i]))
            if predict == self.testy[i]:
                accuracy = accuracy + 1
        accuracy = accuracy/len(self.testX)*100
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

class Layer:






    def sigmoidMatrix(self,alpha,xMatrix):
        x = [i * -alpha for i in xMatrix]
        x = np.exp(x)
        x = 1 + x
        x = 1.0 / x
        return x

    def sigmoid(self,alpha, x):
        return 1.0 / (1 + np.exp(-alpha * x))

    def sigmoidDiff(self,alpha, x):
        return alpha*self.sigmoid(alpha,x)*(1-self.sigmoid(alpha,x))

    def __repr__(self):
        str = "L: %d\n"%self.L
        str = str + "layerInfo"+ repr(self.layerInfo)+'\n'
        # str = str + "X.shape: "+ repr(self.dataset.shape) + '\n'
        str = str + "X: \n"+ repr(self.X) + '\n'
        str = str + "mu: "+ repr(self.learnRate) + '\n'
        str = str + "alpha: "+ repr(self.alpha)+'\n'
        str = str + "N: "+ repr(self.N)+'\n'
        str = str + "weights: \n"+ repr(self.weights)+'\n'
        str = str + "nets: \n"+ repr(self.nets)+'\n'
        str = str + "outputs: \n"+ repr(self.outputs)+'\n'
        str = str + "deltas: \n"+ repr(self.deltas)+'\n'
        return str

    def predict(self,x):
        print('predict ');print(x)
        self.forwardComputation(0,x)
        print('predicted ');print(self.outputs[self.L-1])
        predicted = self.outputs[self.L-1].index(max(self.outputs[self.L-1]))
        print(predicted)
        return predicted

    def testModel(self,X,y):
        accuracy = 0
        for i in range(len(X)):
            blockPrint()
            predicted = self.predict(X[i,:])
            enablePrint()
            print('predicted '+repr(predicted)+' true '+repr(y[i]))
            if predicted == y[i]:
                accuracy = accuracy+1
        accuracy = (accuracy/len(X))*100;
        print('accuracy: '+repr(accuracy))



    def print(self):
        print()
        print('weights');print(self.weights)
        print('nets');print(self.nets)
        print('outputs');print(self.outputs)
        print('deltas');print(self.deltas)

def testModel3():
    url = "G:\Projects\PycharmProjects\PatternRecognition\BackPropagation\wheat-seeds.csv"
    dataset = pandas.read_csv(url)
    array = dataset.values
    X = array[:, 0:7]
    Y = array[:, 7]
    Y = np.subtract(Y,1)
    validation_size = 0.20
    # dataset = [[2.7810836, 2.550537003, 0],
    #            [1.465489372, 2.362125076, 0],
    #            [3.396561688, 4.400293529, 0],
    #            [1.38807019, 1.850220317, 0],
    #            [3.06407232, 3.005305973, 0],
    #            [7.627531214, 2.759262235, 1],
    #            [5.332441248, 2.088626775, 1],
    #            [6.922596716, 1.77106367, 1],
    #            [8.675418651, -0.242068655, 1],
    #            [7.673756466, 3.508563011, 1]]
    # dataset = [
    #     [0, 0, 0],
    #     [0, 1, 1],
    #     [1, 0, 1],
    #     [1, 1, 0],
    #     [0, 0, 0],
    #     [0, 1, 1],
    #     [1, 0, 1],
    #     [1, 1, 0]
    # ]
    # dataset = np.array(dataset)
    #
    # array = dataset
    # X = array[:, 0:2]
    # Y = array[:, 2]
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
    nHidden = 5
    hiddenLayerInfo = [2,2,2,2,2]
    alpha = 1.0
    learningRate = 0.3
    # print(X)

    # NeuralNetwork(nHidden, hiddenLayerInfo, X, Y, None, None, learnRate=learningRate,alpha=alpha, nLoop=100)
    NeuralNetwork(nHidden, hiddenLayerInfo, X_train, Y_train, X_validation, Y_validation, learnRate=learningRate,alpha=alpha, nLoop=100)

dataset = [
    [0, 0, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0]
]
nHidden = 1
hiddenLayerInfo = [5]
alpha = 1.0
learningRate = 0.3
dataset = np.array(dataset)
NeuralNetwork(nHidden, hiddenLayerInfo, dataset[:,0:2], dataset[:,2], dataset[:,0:2], dataset[:,2], learningRate,1.0, nLoop=5000)

