from KMeansClustering import KMeansClustering
from GaussianMixtureModel import GaussianMixtureModel
from sklearn.utils import shuffle
from scipy.stats import zscore

import pandas as pd


def LoadData(path):
    dataSet = pd.read_csv(filepath_or_buffer=path, header=None)
    dataSet = dataSet.fillna(0)
    dataSet.iloc[:, :-1] = dataSet.iloc[:, :-1].apply(zscore)
    dataSet = dataSet.fillna(0)
    dataSet = dataSet.drop_duplicates()
    return shuffle(dataSet)


if __name__ == '__main__':

    dermatologyData = LoadData('./dataset/dermatologyData.csv')
    vowelsData= LoadData('./dataset/vowelsData.csv')
    yeastData= LoadData('./dataset/yeastData.csv')
    soybeanData = LoadData('./dataset/soybeanData.csv')
    glassData = LoadData('./dataset/glassData.csv')
    ecoliData = LoadData('./dataset/ecoliData.csv')

    print('K-Means Dermatology Data Dataset')
    dermatologyKMeansClustering = KMeansClustering(dermatologyData,'Dermatology', range(1, 13))
    dermatologyKMeansClustering.validate()

    print('K-Means Vowels Data Dataset')
    vowelsKMeansClustering = KMeansClustering(vowelsData, 'Vowels',range(1, 23))
    vowelsKMeansClustering.validate()

    print(' K-Means Glass Data Dataset')
    glassKMeansClustering = KMeansClustering(glassData, 'Glass', range(1, 13))
    glassKMeansClustering.validate()

    print(' K-Means Ecoli Data Dataset')
    ecoliKMeansClustering = KMeansClustering(ecoliData, 'Ecoli', range(1, 11))
    ecoliKMeansClustering.validate()

    print(' K-Means Yeast Data Dataset ')
    yeastKMeansClustering = KMeansClustering(yeastData, 'Yeast', range(1, 19))
    yeastKMeansClustering.validate()

    print(' K-Means Soybean Data Dataset')
    soybeanKMeansClustering = KMeansClustering(soybeanData, 'Soybean',range(1, 31))
    soybeanKMeansClustering.validate()


    print ('----------------------------------------------------------------------')


    print('GMM Dermatology Data Dataset')
    dermatologyGMMClustering = GaussianMixtureModel(dermatologyData, 'Dermatology',range(1, 13))
    dermatologyGMMClustering.validate()

    print('GMM Vowels Data Dataset')
    vowelsGMMClustering = GaussianMixtureModel(vowelsData, 'Vowels', range(1, 23))
    vowelsGMMClustering.validate()

    print('GMM Glass Data Dataset')
    glassGMMClustering = GaussianMixtureModel(glassData, 'Glass', range(1, 13))
    glassGMMClustering.validate()

    print('GMM Ecoli Data Dataset')
    ecoliGMMClustering = GaussianMixtureModel(ecoliData, 'Ecoli', range(1, 11))
    ecoliGMMClustering.validate()

    print('GMM Yeast Data Dataset')
    yeastGMMClustering = GaussianMixtureModel(yeastData, 'Yeast', range(1, 19))
    yeastGMMClustering.validate()

    print('GMM Soybean Data Dataset')
    soybeanGMMClustering = GaussianMixtureModel(soybeanData, 'Soybean', range(1, 31))
    soybeanGMMClustering.validate()
