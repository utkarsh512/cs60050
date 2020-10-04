from datetime import date
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

EPOCH = date(2020, 1, 22)

class Record:
    '''object for records'''
    def __init__(self, record):
        '''
        :param record: a comma-separated string representing a record
        '''
        record = record.strip().split(',')
        self.date = genDaysSinceEpoch(record[0])
        self.confirmed = int(record[1])
        self.recovered = int(record[2])
        self.deaths = int(record[3])
        self.increaseRate = float(record[4])

    def get(self):
        return self.__dict__

def genDaysSinceEpoch(cur_date):
    '''
    Utility to convert a date to no. of days since EPOCH
    :param cur_date: a string containing date, format is as per the datset
    :return: no. of days since EPOCH
    '''
    cur_date = tuple(int(x) for x in cur_date.split('/'))
    m, d, y = cur_date
    return (date(y, m, d) - EPOCH).days

def squared_residual_sum(y):
    '''
    Utitlity to evaluate Squared Residual Sum for a numpy array
    :param y: 1-dimensional numpy array
    :return: its Squared Residual Sum
    '''
    return np.sum((y - np.mean(y)) ** 2)

def rss(y_left, y_right):
    return squared_residual_sum(y_left) + squared_residual_sum(y_right)

def split(X_train, y_train, depth, max_depth):
    '''
    Utility to split the instances by choosing best attribute and building a tree
    :param X_train: training instances
    :param y_train: training labels
    :param depth: current depth of the tree
    :param max_depth: maximum allowed depth
    :return: a decision tree
    '''
    if depth == max_depth or len(X_train) < 2:
        return {'prediction': np.mean(y_train)}

    attr = findBestAttribute(X_train, y_train)
    left_idx = X_train[attr['feature']] < attr['threshold']
    attr['left'] = split(X_train[left_idx], y_train[left_idx],
                         depth + 1, max_depth)
    attr['right'] = split(X_train[~left_idx], y_train[~left_idx],
                          depth + 1, max_depth)

    attr['_prediction'] = np.mean(y_train)
    return attr

def findBestAttribute(X_train, y_train):
    '''
    Utility to find the best attribute for any split. Uses
    Residual Squared Sum as deciding strategy
    :param X_train: training instances
    :param y_train: training labels
    :return: best attribute to split
    '''
    best_feature, best_threshold, min_rss = None, None, np.inf
    for feature in X_train.columns:
        thresholds = X_train[feature].unique().tolist()
        thresholds.sort()
        thresholds = thresholds[1:]
        for t in thresholds:
            y_left_idx = X_train[feature] < t
            y_left = y_train[y_left_idx]
            y_right = y_train[~y_left_idx]
            t_rss = rss(y_left, y_right)
            if t_rss < min_rss:
                min_rss = t_rss
                best_threshold = t
                best_feature = feature
    return {'feature': best_feature, 'threshold': best_threshold}

def predict(sample, tree):
    '''
    Utility to predict using generated decision tree
    :param sample: instance to predict
    :param tree: generated decision tree
    :return: prediction for the instance
    '''
    prediction = None
    while prediction is None:
        feature, threshold = tree['feature'], tree['threshold']
        if sample[feature] < threshold:
            tree = tree['left']
        else:
            tree = tree['right']
        prediction = tree.get('prediction', None)
    return prediction

def evaluate(X, y, tree):
    y_preds = X.apply(predict, axis='columns', tree=tree.copy())
    mse = np.sum((y - y_preds) ** 2)
    mse /= X.shape[0]
    return mse

def getXYFromDataframe(df):
    '''
    Utility to generate feature vectors and labels from pandas Dataframe object
    :param df: pandas Dataframe object
    :return: feature vectors and labels
    '''
    X = df[['date', 'confirmed', 'recovered', 'deaths']]
    y = df[['increaseRate']].to_numpy().squeeze()
    return X, y

def buildTree(df, max_depth):
    '''
    Utility to build a decision tree for given max_depth from the instances
    in the df dataframe. 10 random splits are considered to get trainSet, validationSet and testError.
    Best of them is returned
    :param df: pandad Dataframe object containing dataset
    :param max_depth: maximum allowed depth
    :return: best tree with given maximum depth, and all the errors associated with it
    '''
    idx1 = int(df.shape[0] * 0.6)
    idx2 = int(df.shape[0] * 0.8)
    bestTree = None
    bestTreeTrainDf = None
    bestTreeTestDf = None
    bestTreeValDf = None
    trainError, testError, valError = [], [], []
    bestError = np.inf

    for _ in range(10):
        newdf = df.copy()
        newdf = newdf.iloc[np.random.permutation(len(newdf))]
        traindf = newdf[:idx1]
        valdf = newdf[idx1:idx2]
        testdf = newdf[idx2:]
        X_train, y_train = getXYFromDataframe(traindf)
        X_test, y_test = getXYFromDataframe(testdf)
        X_val, y_val = getXYFromDataframe(valdf)
        tree = split(X_train, y_train, 0, max_depth)
        curTrainError = evaluate(X_train, y_train, tree)
        curTestError = evaluate(X_test, y_test, tree)
        curValError = evaluate(X_val, y_val, tree)
        trainError.append(curTrainError)
        testError.append(curTestError)
        valError.append(curValError)
        if (curTestError < bestError):
            bestError = curTestError
            bestTree = tree
            bestTreeTrainDf = traindf
            bestTreeTestDf = testdf
            bestTreeValDf = valdf

    return bestTree, testError, trainError, valError, max_depth, bestTreeTrainDf, bestTreeTestDf, bestTreeValDf

def canPrune(tree, X, y):
    '''
    Utility to check whether passed node can be pruned or not
    :param tree: Node to prune
    :param X: feature vectors
    :param y: feature labels
    :return: True/False as per the decision
    '''
    if len(X) == 0:
        return True
    y_pred = tree['_prediction']
    mse = np.sum((y - y_pred) ** 2)
    mse /= len(X)
    if mse <= evaluate(X, y, tree):
        return True
    return False

def pruneTreeUtil(tree, X, y):
    '''
    Utility to prune Decision trees recursively
    :param tree: tree to prune
    :param X: feature vectors
    :param y: feature labels
    :return: None
    '''
    if 'prediction' in tree.keys():
        return
    if len(X) == 0:
        tree['prediction'] = tree['_prediction']
        return
    indexes = X[tree['feature']] < tree['threshold']
    pruneTreeUtil(tree['left'], X[indexes], y[indexes])
    pruneTreeUtil(tree['right'], X[~indexes], y[~indexes])
    if canPrune(tree, X, y):
        tree['prediction'] = tree['_prediction']

def printTree(tree, level=0, conditional='if '):
    '''
    Utility to print the decision tree as if-then-else statements
    '''
    if 'prediction' in tree.keys():
        conditional = conditional.replace('if ', '')
        ret = '\t' * level + conditional + 'increaseRate = ' + str(tree['prediction']) + '\n'
        return ret
    ret = '\t' * level + conditional + tree['feature']
    ret += ' < ' + str(tree['threshold']) + '\n'
    ret += printTree(tree['left'], level + 1, 'then if ')
    ret += printTree(tree['right'], level + 1, 'else if ')
    return ret

def visualise(trainError, testError, xlabel, ylabel, title, saveAs):
    '''
    Utility to plot trainError and testError
    '''
    plt.switch_backend('Agg')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(trainError, label='Training Error')
    plt.plot(testError, label='Testing Error')
    plt.title(title)
    plt.legend()
    plt.savefig(saveAs)
    plt.close()

def analyzeTree(treeData, dS):
    dS.write('Analyzing passed tree:\n')
    _, testError, trainError, _valError, max_depth, _trainDf, _testDf, _valDf = treeData
    dS.write('\tmax_depth = {0}'.format(max_depth))
    for i in range(len(testError)):
        dS.write('\tVariation: {0} -> trainError = {1}, testError = {2}\n'.format(i,trainError[i], testError[i]))
    dS.write('\tmeanTrainError = {0}, meanTestError = {1}\n\n'.format(np.mean(trainError),
                                                                  np.mean(testError)))
    visualise(trainError, testError,
              xlabel='Variations for same depth',
              ylabel='Mean Squared Error',
              title='Mean Squared Error of variations of same maximum depth',
              saveAs='tree.png')

def analyzeModel(df, dS):
    trainError = [None]
    testError = [None]
    bestDepth = None
    bestError = np.inf
    dS.write('Analyzing the decision tree model:\n')
    for max_depth in range(1, 21):
        _, curTestError, curTrainError, _curValError, _max_depth, _trainDf, _testDf, _valDf = buildTree(df, max_depth)
        curTrainError = np.mean(curTrainError)
        curTestError = np.mean(curTestError)
        trainError.append(curTrainError)
        testError.append(curTestError)
        if (curTestError < bestError):
            bestError = curTestError
            bestDepth = max_depth
        dS.write('\tmax_depth = {0} -> trainError = {1}, testError = {2}\n'.format(max_depth,
                                                                                   curTrainError,
                                                                                   curTestError))
    dS.write('\n')
    visualise(trainError, testError,
              xlabel='Maximum Depth',
              ylabel='Mean Squared Error',
              title='Mean Squared Error of the decision tree for different maximum depths',
              saveAs='analyze.png')
    return bestDepth

if __name__ == '__main__':
    iF = open('PercentageIncreaseCOVIDWorldwide.csv', 'r')
    dS = open('data.txt', 'w')
    tS = open('tree.txt', 'w')
    pS = open('prunedTree.txt', 'w')
    records = [x.strip() for x in iF.readlines()][2:]
    records = [Record(x).get() for x in records]
    df = pd.DataFrame(records)

    # Question 1
    dS.write('Solution to Question 1\n')
    max_depth = int(input('Enter maximum depth: '))

    # As the maximum depth of tree is not bounded, we are using maximum depth as 10
    if max_depth == -1:
        max_depth = 10 
    treeData = buildTree(df, max_depth)
    analyzeTree(treeData, dS)

    # Question 2
    dS.write('Solution to Question 2\n')
    best_depth = analyzeModel(df, dS)

    # Question 3
    dS.write('Solution to Question 3\n')
    treeData = buildTree(df, best_depth)
    tree, _testError, _trainError, _valError, _max_depth, traindf, testdf, valdf = treeData
    X_train, y_train = getXYFromDataframe(traindf)
    X_test, y_test = getXYFromDataframe(testdf)
    X_val, y_val = getXYFromDataframe(valdf)
    tS.write(printTree(tree))
    dS.write('\tBefore pruning:\n')
    dS.write('\t\ttrainError = {0}, valError = {1}, testError = {2}\n'.format(evaluate(X_train, y_train, tree),
                                                                            evaluate(X_val, y_val, tree),
                                                                            evaluate(X_test, y_test, tree)))
    pruneTreeUtil(tree, X_val, y_val)
    pS.write(printTree(tree))
    dS.write('\tAfter pruning:\n')
    dS.write('\t\ttrainError = {0}, valError = {1}, testError = {2}\n'.format(evaluate(X_train, y_train, tree),
                                                                            evaluate(X_val, y_val, tree),
                                                                            evaluate(X_test, y_test, tree)))
    iF.close()
    dS.close()
    tS.close()
    pS.close()
