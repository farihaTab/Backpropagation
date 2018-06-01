import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
import numpy as np
import random
import sys, os
from csv import reader

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

def forwardPropagate(row,weights,bias,layerInfo,L):
    outputs = [[] for _ in range(L)]
    outputs[0] = row
    for r in range(1,L):
        # print("r: " + repr(r) + 'th layer')
        currNeurons = layerInfo[r]
        prevNeurons = layerInfo[r - 1]
        input = outputs[r - 1]
        # input.append(1) #for bias
        for j in range(currNeurons):
            try:
                weight = weights[r][j]
            except:
                print(r);print(j)
            output = calcOutput(input,weight,bias[r])
            outputs[r].append(output)
        # print('output '+repr(outputs[r]))
    return outputs
def calcOutput(input,weight,bias):
    net = bias
    for k in range(len(weight)):
        net += weight[k]*input[k]
    # print('net '+repr(net))
    output = sigmoid(net)
    return output
def backwardPropagate(outputs,expected,bias,layerInfo,L):
    deltas = [[] for _ in range(L)]
    for r in range(L-1,0,-1):
        # print("r: " + repr(r) + 'th layer')
        currNeurons = layerInfo[r]
        prevNeurons = layerInfo[r - 1]
        if r == L-1:
            for j in range(currNeurons):
                # print('j: '+repr(j)+'th neuron')
                out = outputs[r][j]
                target = expected[j]
                delta = -(target-out)*out*(1-out)
                # print('delta ='+repr(delta)+ '= -(target-out)*out*(1-out) ='+repr(-(target-out))+'*'+repr(out*(1-out)))
                # print('check ')
                # for k in range(prevNeurons):
                    # print('\tw_jk changed = '+repr(weights[r][j][k]-0.5*delta*outputs[r-1][k])+'= w_jk - 0.5*(-delta*out_prev)='+repr(weights[r][j][k])+'-0.5*'+repr(-delta*outputs[r-1][k]))
                deltas[r].append(delta)
        else:
            nextNeurons = layerInfo[r+1]
            for j in range(currNeurons):
                delta = calcDelta(nextNeurons,deltas[r+1],weights[r+1],outputs[r][j],j)
                # print('delta '+repr(delta))
                # print('check ')
                # for k in range(prevNeurons):
                #     print('\tw_jk changed = ' +
                #           repr(weights[r][j][k] - 0.5 * delta * outputs[r - 1][k]) + '= w_jk - 0.5*(-delta*out_prev)=' + repr(
                #         weights[r][j][k]) + '-0.5*' + repr(delta * outputs[r - 1][k]))
                deltas[r].append(delta)
    return deltas
def calcDelta(nextNeurons,nextDeltas,weights,output,j):
    s = 0.0
    for k in range(nextNeurons):
        s = s+nextDeltas[k]*weights[k][j]
    d = s*output*(1-output)
    return d
def calculateDelWeight(delWeights,deltas,outputs,layerInfo,L):
    for r in range(1,L):
        # print("r: " + repr(r) + 'th layer')
        currNeurons = layerInfo[r]
        prevNeurons = layerInfo[r - 1]
        for j in range(currNeurons):
            for k in range(prevNeurons):
                delWeights[r][j][k] = delWeights[r][j][k] - deltas[r][j]*outputs[r-1][k]
                # print('∆w_' + repr(j) + repr(k) +': '+repr(- deltas[r][j]*outputs[r-1][k]))
    return delWeights
def updateWeights(delWeights,weights,learning_rate,layerInfo,L):
    for r in range(L-1,0,-1):
        print("r: " + repr(r) + 'th layer')
        currNeurons = layerInfo[r]
        prevNeurons = layerInfo[r - 1]
        for j in range(currNeurons):
            for k in range(prevNeurons):
                old = weights[r][j][k]
                weights[r][j][k] = weights[r][j][k] + learning_rate*delWeights[r][j][k]
                # print('w_jk(new) ='+repr(weights[r][j][k])
                #       +'= w_jk(old)+mu*∆w_' + repr(j) + repr(k) + '= '
                #       +repr(old)+'+'+repr(learning_rate)+'*'+repr(delWeights[r][j][k]))
                print('∆w_' + repr(j) + repr(k) + ' ' + repr(delWeights[r][j][k]) + ' old ' + repr(old)
                      + ' new ' + repr(weights[r][j][k]))
def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))
def calculate_total_error(training_sets,weights, bias, layerInfo, L):
    total_error = 0
    for t in range(len(training_sets)):
        training_inputs, training_outputs = training_sets[t]
        outputs = forwardPropagate(training_inputs, weights, bias, layerInfo, L)
        output = outputs[-1]
        for o in range(len(training_outputs)):
            total_error += 0.5*(training_outputs[o]-output[o])**2
        predicted = output.index(max(output))
        actual = training_outputs.index(max(training_outputs))
        print(predicted, actual)
    return total_error
learning_rate = 0.01
L = 3
layerInfo = [2,2,2]
weights = [[],[[0.15,0.20],[.25,.3]],[[.4,.45],[.5,.55]]]

bias = [0,0.35,0.6]

# row = [0.05,0.1]
# expected = [0.01,0.99]
# print('\nforward pass')
# outputs = forwardPropagate(row,weights,bias,layerInfo,L)
# print('hidden outputs ');print(outputs[1])
# print('outer outputs ');print(outputs[2])
# blockPrint()
# print('\ncalculate error')
# deltas = backwardPropagate(outputs,expected,bias,layerInfo,L)
# enablePrint()
# print('output neuron deltas ');
# print(deltas[2])
# print('hidden neuron deltas ');
# print(deltas[1])
# print('\ncalculate change to make')
# delWeights = [[], [[0.0, 0.0], [.0, .0]], [[.0, .0], [.0, .0]]]
# delWeights = calculateDelWeight(delWeights,deltas,outputs,layerInfo,L)
# print('update weights')
# updateWeights(delWeights,weights,learning_rate,layerInfo,L)
training_sets = [
    [[100, 0], [1, 0]],
    [[0, 100], [0, 1]],
    [[100, 0], [0, 1]],
    [[100, 100], [1, 0]]
]
# training_sets = [
#     [[0.05, 0.1], [0.01, 0.99]],
#     [[0.1, 0.05], [0.99, 0.01]]
# ]
for i in range(1):
    for  j in range(len(training_sets)):
        row = training_sets[j][0]
        expected = training_sets[j][1]
        print('\nforward pass')
        outputs = forwardPropagate(row,weights,bias,layerInfo,L)
        print('hidden outputs ');print(outputs[1])
        print('outer outputs ');print(outputs[2])
        blockPrint()
        print('\ncalculate error')
        deltas = backwardPropagate(outputs,expected,bias,layerInfo,L)
        enablePrint()
        print('output neuron deltas ');
        print(deltas[2])
        print('hidden neuron deltas ');
        print(deltas[1])
        print('\ncalculate change to make')
        delWeights = [[], [[0.0, 0.0], [.0, .0]], [[.0, .0], [.0, .0]]]
        delWeights = calculateDelWeight(delWeights,deltas,outputs,layerInfo,L)
        print('update weights')
        updateWeights(delWeights,weights,learning_rate,layerInfo,L)

    # print(round(calculate_total_error([[[0.05, 0.1], [0.01, 0.99]]],weights,bias,layerInfo,L),9))
    print(i, round(calculate_total_error(training_sets,weights,bias,layerInfo,L),9))
