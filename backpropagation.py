import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import numpy as np
import random
import sys, os

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

# X = (hours sleeping, hours studying), y = score on test
# X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
# y = np.array(([0], [1], [1]), dtype=float)
Xx = np.array(([[0.05,0.10],[0.1,0.05]]), dtype=float)
yy = np.array(([[1],[0]]), dtype=float)

class NeuralNetwork:
    def __init__(self, L, layerInfo, X,y, mu, alpha):
        self.L = L
        self.layerInfo = layerInfo
        # self.dataset = dataset #dataset
        self.mu = mu #learning rate
        self.alpha = alpha
        self.N = X.shape[0]
        self.weights = []
        self.nets = []
        self.outputs = []
        self.deltas = []
        self.X = X
        self.y = y
        self.trueY = [[] for _ in range(self.N)]


    def configure(self):
        # update X as a concatenation of dataset and 1
        # w = np.array(([[1.0] * self.N]))
        # w = w.T
        # print(w)
        # self.X = np.concatenate((self.dataset, w), 1)
        # print(self.X)
        # self.X = self.dataset
        # self.y = self.dataset[:,self.layerInfo[self.L]]
        # initialize weights
        # rand = np.random.normal(0.0,0.01,1000)
        seed = 7
        np.random.seed(seed)
        random.seed(seed)
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

    def test(self):
        self.weights = []
        self.weights.append(np.array([[0.2,0.3,0.4],[0.3,0.4,0.5],[0.4,0.5,0.6]]))
        self.weights.append(np.array([[0.3,0.4,0.5,0.6],[0.4,0.5,0.6,0.7]]))
        print(self.weights)




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

    def updateWeights(self):
        None

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
                sum = np.multiply(-self.mu,sum)
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
        str = str + "mu: "+ repr(self.mu) +'\n'
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

def testModel():
    L = 2
    # layerInfo = [2,3,4,2]
    layerInfo = [2, 3, 2]
    nn = NeuralNetwork(L, layerInfo, Xx, yy, 1.0, 1.0)
    nn.configure();
    for i in range(2):
        print('train ' + repr(i))
        nn.train()
    nn.predict([0.5, 1.0])
    nn.predict([0.5, 0.01])
    nn.predict([0.01, 0.5])

def testModel2():
    url = "G:\Projects\PycharmProjects\PatternRecognition\BackPropagation\iris.data"
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    dataset = pandas.read_csv(url, names=names)

    print(dataset.groupby('class').size())
    classes = pandas.unique(dataset['class'].values.ravel('K'))
    i = 0
    for cls in classes:
        dataset = dataset.replace(cls,i)
        i = i+1

    array = dataset.values
    X = array[:, 0:4]
    Y = array[:, 4]
    # print(Y)
    validation_size = 0.20
    seed = 7
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size,
                                                                                    random_state=seed)
    print(X_train.shape)
    print(X_validation.shape)
    # print(Y_train.shape)
    L = 3
    layerInfo = [4, 8, 8, 3]
    nn = NeuralNetwork(L, layerInfo, X_train, Y_train, 0.5, 1.0)
    nn.configure()
    for i in range(100):
        print('\n\n\nTRAINING '+repr(i))
        blockPrint()
        nn.train()
        enablePrint()
    nn.testModel(X_validation,Y_validation)

def testModel3():
    url = "G:\Projects\PycharmProjects\PatternRecognition\BackPropagation\wheat-seeds.csv"
    dataset = pandas.read_csv(url)
    print(dataset)

    array = dataset.values
    X = array[:, 0:7]
    Y = array[:, 7]
    # print(Y)
    validation_size = 0.20
    seed = 7
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size,
                                                                                    random_state=seed)
    L = 3
    layerInfo = [7, 8, 8, 4]
    nn = NeuralNetwork(L, layerInfo, X_train, Y_train, 0.5, 1.0)
    nn.configure()
    for i in range(10):
        print('\n\n\nTRAINING '+repr(i))
        blockPrint()
        nn.train()
        enablePrint()
    nn.testModel(X_validation,Y_validation)

def takeInput():
    L = int(input("Enter number of hidden layers: "))
    hlayerInfo = []
    for i in range(L):
        msg = "no of neurons in %dth  hidden layer:" % (i+1)
        hlayerInfo.append(int(input(msg)))
    return L,hlayerInfo
def main():
    print('hello')
    # L = int(input("Enter L: "))
    # layerInfo = []
    # for i in range(L+1):
    #     msg = "no of neurons in %dth layer:" % i
    #     layerInfo.append(int(input(msg)))
    # testModel()
    # testModel3()
    takeInput()


if __name__ == "__main__":
    main()

