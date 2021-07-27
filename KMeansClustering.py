from __future__ import division
from copy import deepcopy

import numpy as np
import collections
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class KMeansClustering:
    def __init__(self,data,pltName,K=range(1, 11),tolerance=0.00001,epochs=100):
        self.data = data
        self.pltName = pltName
        self.K = K
        self.tolerance = tolerance
        self.epochs = epochs

    def GetDistance(self, a, b, ax=1):
        distance=np.linalg.norm(a - b, axis=ax)
        return distance

    def GetSSE(self, X, clusters, C):
        sse = 0
        for i in range(len(X)):
            sumvalue=np.sum(np.square(np.subtract(X[i], C[int(clusters[i])])))
            sse =sse+sumvalue
        return sse

    def GetNMI(self, Y, clusters, C):
        labelCounter = collections.Counter(Y)
        clusterCounter = collections.Counter(clusters)
        Probabilities = {}
        hy = 0
        hc = 0
        hyc = 0

     
        for label in labelCounter:
            labelProb = labelCounter[label] / len(Y)
            if labelProb > 0:
                hy =hy- labelProb * math.log(labelProb, 2)

        
       
        for c in range(len(C)):
            Prob = clusterCounter[c] / len(clusters)
            Probabilities[c] = Prob
            if Prob > 0:
                hc =hc- Prob * math.log(Prob, 2)

       
        for c in range(len(C)):
            classLabels = [Y[j] for j in range(len(Y)) if clusters[j] == c]
            classLabelCounter = collections.Counter(classLabels)
            nhy = 0
            for label in classLabelCounter:
                labelProb = classLabelCounter[label] / len(classLabels)
                if labelProb > 0:
                    nhy += labelProb * math.log(labelProb, 2)
            hyc =hyc- Probabilities[c] * nhy
        
        iyc = hy - hyc
        return 2 * iyc / (hy + hc)

    def validate(self):
        SSElist = []
        NMIlist = []

        X = self.data.iloc[:, :-1].values
        Y = self.data.iloc[:, -1].values

        print("K\tSSE\tNMI")
        for k in self.K:
            C = X[np.random.choice(len(X), k, False), :]
            cOld = np.zeros(C.shape)
            clusters = np.zeros(len(X))

            error = self.tolerance + 1
            for _ in range(self.epochs):
                if error > self.tolerance:
                    for i in range(len(X)):
                        clusters[i] = np.argmin(self.GetDistance(X[i], C))
                    cOld = deepcopy(C)
                    for i in range(k):
                        points = [X[j] for j in range(len(X)) if clusters[j] == i]
                        if (len(points) > 0):
                            C[i] = np.mean(points, axis=0)
                    error = self.GetDistance(C, cOld, None)
                else:
                    break
            sseValue = self.GetSSE(X, clusters, C)
            nmiValue = self.GetNMI(Y, clusters, C)

            print("{}\t{}\t{}".format(k, sseValue, nmiValue))

            SSElist.append(sseValue)
            NMIlist.append(nmiValue)

        plt.figure()
        plt.plot(self.K, SSElist, 'ro-')
        plt.ylabel('y:SSE')
        plt.xlabel('x:K')
        plt.title("{}  K VS SSE".format(self.pltName))
        plt.savefig("KMeans Clustering SSE {}".format(self.pltName))
