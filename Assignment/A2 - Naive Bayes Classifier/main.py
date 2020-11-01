# -*- coding: utf-8 -*-

'''
Source code for the Machine Learning Assignment 2 for course CS60050 IIT Kharagpur

Group: 9
Group Members:
    Utkarsh Patel 18EC30048
    Akhil 18CS10070

Naive Bayes Classifier for estimating stay of a patient in a hospital.
'''

import numpy as np
import math
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

DATA_FILE_ = 'Train_B.csv'  # dataset
K_ = 5                      # K-fold cross validation
PARTITION = 0.8             # PARTITIONING SCHEME
BREAK_ = '-' * 50 + '\n'

def getXY(df):
    '''
    Utility to generate feature vectors and feature labels from pandas DataFrame object
    :param df: Dataset as pandas DataFrame object
    :return: feature vectors and feature labels as pandas DataFrame object
    '''
    columns = list(df.columns)
    columns.remove('Stay')
    X = df[columns]
    y = df[['Stay']]
    return X, y

def getXYArray(df):
    '''
    Similar to getXY() method, but outputs are numpy arrays not pandas DataFrame objects
    :param df: Dataset as pandas DataFrame object
    :return: feature vectors and feature labels as numpy arrays
    '''
    X, y = getXY(df)
    X = X.to_numpy()
    y = y.to_numpy().squeeze()
    return X, y

def visualize(components, xlabel, ylabel, title, saveAs):
    '''
    Utility to plot graphs and save it on machine
    :param components: List of Python dictionaries containing components to plot
    :param xlabel: Label on the horizontal axis
    :param ylabel: Label on the vertical axis
    :param title: Title of the plot
    :param saveAs: Name of the plot
    :return: creates the graph, and saves it on the machine
    '''
    plt.switch_backend('Agg')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    for c in components:
        plt.plot(c['data'], label=c['label'])
    plt.title(title)
    plt.legend()
    plt.savefig(saveAs)
    plt.close()

class NaiveBayes:
    '''
    Naive Bayes classifier for dataset containing categorical values only
    '''
    def __init__(self):
        self.prob = dict()

    def fit(self,X_train, y_train):
        columns = list(X_train)
        for i, row in X_train.iterrows():
            label = y_train.loc[i, 'Stay']
            right = 'Stay={0}'.format(label)
            if right in self.prob.keys():
                self.prob[right] += 1
            else:
                self.prob[right] = 1
            for j in columns:
                val = row[j]
                left = '{0}={1}'.format(j, val)
                left = left + '|'+right
                if left in self.prob.keys():
                    self.prob[left] += 1
                else:
                    self.prob[left] = 1

        for i in self.prob.keys():
            if '|' in i:
                right = i.split('|')[1]
                self.prob[i] /= self.prob[right]

        for i in self.prob.keys():
            if '|' not in i:
                self.prob[i] /= len(X_train)

    def score(self, X_test, y_test, plt_show=False, saveAs=None):
        true_prediction = false_prediction = 0
        columns = list(X_test)
        labels = [x for x in range(11)]
        actual = []
        pred = []
        for i, row in X_test.iterrows():
            label = y_test.loc[i, 'Stay']
            lst = []
            for j in columns:
                val = row[j]
                lst.append('{0}={1}'.format(j, val))
            best = -1
            weight = 0
            for j in labels:
                right = 'Stay={0}'.format(j)
                cur_weight = self.prob[right]
                for it in lst:
                    it = it + '|' + right
                    if it in self.prob.keys():
                        cur_weight *= self.prob[it]
                    else:
                        cur_weight /= self.prob[right] # Laplace correction
                if cur_weight > weight:
                    weight = cur_weight
                    best = j
            actual.append(label)
            pred.append(best)
            if best == label:
                true_prediction += 1
            else:
                false_prediction += 1
        if plt_show:
            self.plot(actual, pred, saveAs)
        return true_prediction / (true_prediction + false_prediction)

    def plot(self, actual, pred, saveAs):
        plt.switch_backend('Agg')
        plt.scatter(actual, pred)
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.title('True V Predicted Value - Scatter Plot')
        plt.savefig(saveAs)
        plt.close()

class NaiveBayesv2:
    '''
    Naive Bayes classifier for continuous-valued attributes
    using Gaussian kernel
    '''
    def __init__(self):
        self.classes = None
        self.class_freq = None
        self.class_prob = None
        self.means = None
        self.std = None

    def fit(self, X, y):
        X_ = self.separate(X, y)
        self.means = {}
        self.std = {}
        for c in self.classes:
            self.means[c] = np.mean(X_[c], axis=0)[0]
            self.std[c] = np.std(X_[c], axis=0)[0]

    def separate(self, X, y):
        self.classes = np.unique(y)
        indexes = {}
        datasets = {}
        cls, counts = np.unique(y, return_counts=True)
        self.class_freq = dict(zip(cls, counts))
        for c in self.classes:
            indexes[c] = np.argwhere(y==c)
            datasets[c] = X[indexes[c], :]
            self.class_freq[c] = self.class_freq[c] / sum(list(self.class_freq.values()))
        return datasets

    def calc_prob(self, x, mean, stdev):
        exponent = math.exp(-((x - mean) ** 2 / (2 * stdev ** 2)))
        return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

    def predict_prob(self, X):
        self.class_prob = {cls:math.log(self.class_freq[cls], math.e) for cls in self.classes}
        for cls in self.classes:
            for i in range(len(X)):
                self.class_prob[cls] += math.log(self.calc_prob(X[i], self.means[cls][i], self.std[cls][i]), math.e)
        self.class_prob = {cls: math.e ** self.class_prob[cls] for cls in self.class_prob}
        return self.class_prob

    def predict(self, X):
        ''' This funtion predicts the class of a sample '''
        pred = []
        for x in X:
            pred_class = None
            max_prob = 0
            for cls, prob in self.predict_prob(x).items():
                if prob > max_prob:
                    max_prob = prob
                    pred_class = cls
            pred.append(pred_class)
        return pred

    def score(self, X, y):
        y_pred = self.predict(X)
        true_prediction = false_prediction = 0
        for i in range(len(y)):
            if y[i] == y_pred[i]:
                true_prediction += 1
            else:
                false_prediction += 1
        return true_prediction / (true_prediction + false_prediction)

def KFold(k, df, saveAs='accuracy.png'):
    '''
    Utility to perform K-Fold cross validation on Discrete Naive Bayes classifier
    :param k: value of parameter K
    :param df: pandas DataFrame object
    :return: Average accuracy after cross validation
    '''
    model_acc = []  # Accuracy observed over our model
    batch = df.shape[0] // k
    mean_acc = 0
    for i in range(k):
        new_df = df.copy()
        test_df = new_df[i * batch: (i + 1) * batch]
        train_df = new_df.drop(test_df.index)
        clf = NaiveBayes()
        X_train, y_train = getXY(train_df)
        X_test, y_test = getXY(test_df)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        mean_acc += score
        model_acc.append(score)
    mean_acc /= k
    plt.switch_backend('Agg')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.plot(model_acc, label='Discrete Naive Bayes')
    plt.axhline(y=mean_acc, linewidth=1, color='k', label='Mean accuracy')
    plt.title('Accuracy Observed During {0}-fold Cross Validation'.format(k))
    plt.legend()
    plt.savefig(saveAs)
    plt.close()
    return mean_acc

def KFoldv2(k, df, pca, saveAs='accuracy.png'):
    '''
    Utility to perform K-Fold cross validation on Naive Bayes classifier with Gaussian kernel
    :param k: value of parameter K
    :param df: pandas DataFrame object
    :return: Average accuracy after cross validation
    '''
    model_acc = []  # Accuracy observed over our model
    batch = df.shape[0] // k
    mean_acc = 0
    for i in range(k):
        new_df = df.copy()
        test_df = new_df[i * batch: (i + 1) * batch]
        train_df = new_df.drop(test_df.index)
        clf = NaiveBayesv2()
        X_train, y_train = getXY(train_df)
        X_test, y_test = getXY(test_df)
        X_train = pca.fit_transform(X_train)
        y_train = y_train.to_numpy().squeeze()
        X_test = pca.fit_transform(X_test)
        y_test = y_test.to_numpy().squeeze()
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        mean_acc += score
        model_acc.append(score)
    mean_acc /= k
    plt.switch_backend('Agg')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.plot(model_acc, label='Gaussian Naive Bayes')
    plt.axhline(y=mean_acc, linewidth=1, color='k', label='Mean accuracy')
    plt.title('Accuracy Observed During {0}-fold Cross Validation'.format(k))
    plt.legend()
    plt.savefig(saveAs)
    plt.close()
    return mean_acc

def train_test(X_train, y_train, X_test, y_test):
    clf = NaiveBayes()
    clf.fit(X_train, y_train)
    return clf.score(X_test, y_test)

def BackwardSelection(X, y):
    '''
    Utility for implementing backward selection method for feature selection
    :param X: Training feature vectors as pandas DataFrame object
    :param y: Training labels as pandas DataFrame object
    :return: best feature subset for describing the data
    '''
    features = ['Hospital_code', 'Hospital_type_code', 'City_Code_Hospital', 'Hospital_region_code',
                'Available Extra Rooms in Hospital', 'Department', 'Ward_Type', 'Ward_Facility_Code',
                'Bed Grade', 'City_Code_Patient', 'Type of Admission', 'Severity of Illness',
                'Visitors with Patient', 'Age', 'Admission_Deposit']

    # We first need to create a validation set (80-20) split
    PARTITION = int(0.8 * X.shape[0])
    X_train = X[:PARTITION]
    y_train = y[:PARTITION]
    X_val = X[PARTITION:]
    y_val = y[PARTITION:]
    observed_accuracy = []
    accuracy = train_test(X_train, y_train, X_val, y_val)
    observed_accuracy.append(accuracy)
    candidate = None # feature to be removed
    it = 0
    while True:
        it += 1
        print("-"*20)
        print("Iteration {0}".format(it))
        print("-"*20)
        for feature in features:
            print("Feature selected: {0}".format(feature))
            X_train_ = X_train.drop([feature], axis=1)
            X_val_ = X_val.drop([feature], axis=1)
            cur_accuracy = train_test(X_train_, y_train, X_val_, y_val)
            print("Observed ac = {0}, Best ac = {1}".format(cur_accuracy, accuracy))
            if cur_accuracy > accuracy:
                accuracy = cur_accuracy
                candidate = feature

        if candidate is None:
            break
        X_train = X_train.drop([candidate], axis=1)
        X_val = X_val.drop([candidate], axis=1)
        features.remove(candidate)
        observed_accuracy.append(accuracy)
        candidate = None
    plt.switch_backend('Agg')
    plt.xlabel('# features removed')
    plt.ylabel('Accuracy')
    plt.plot(observed_accuracy, label='Observed Accuracy')
    plt.title('Observed Accuracy in Backward-selection method')
    plt.legend()
    plt.savefig('plot_3b_accuracy_backselection.png')
    plt.close()
    return features


if __name__ == '__main__':
    data = pd.read_csv(DATA_FILE_)
    writer = open('output_detailed.txt', 'w')
    op = open('output.txt', 'w')
    writer.write('CS60050 Assignment 2: Naive Bayes Classifier\n' + BREAK_)
    writer.write('DATASET\n' + str(data) + '\n\n')

    writer.write('\n\n' + BREAK_ + 'Question 1\n' + BREAK_)

    writer.write('Part a\n' + BREAK_)
    nanCount = len(data) - data.count()
    writer.write('\nMissing values in various columns:\n')
    writer.write('\n' + str(nanCount) + '\n')
    dialog = '\nIt has been observed that missing values exist only in columns \'Bed Grade\' and \'City_Code_Patient\'.\nMost common class of each column will be used to fill up the missing values.\n'
    data['Bed Grade'] = data['Bed Grade'].fillna(data['Bed Grade'].value_counts().index[0])
    data['Bed Grade'] = data['Bed Grade'].astype(np.int64)
    data['City_Code_Patient'] = data['City_Code_Patient'].fillna(data['City_Code_Patient'].value_counts().index[0])
    data['City_Code_Patient'] = data['City_Code_Patient'].astype(np.int64)
    writer.write(dialog)
    writer.write('\nLet\'s see if the missing values have been filled or not\n')
    nanCount = len(data) - data.count()
    writer.write('\n' + str(nanCount) + '\n')
    writer.write('\n' + BREAK_)

    writer.write('Part b\n' + BREAK_)
    dialog = '\nTime to find which columns contain categorical data. For these columns, the datatype would be \'object\'.\n'
    writer.write(dialog)
    writer.write('\n' + str(data.dtypes) + '\n')
    dialog = '\nFor the columns which have datatype as \'object\' will be encoded with sklearn LabelEncoder API.\n'
    writer.write(dialog)
    columns = ['Hospital_region_code', 'Department', 'Ward_Type',
               'Ward_Facility_Code', 'Type of Admission',
               'Severity of Illness', 'Age', 'Stay', 'Hospital_type_code']
    for c in columns:
        data[c] = LabelEncoder().fit_transform(data[c])
    dialog = '\nAfter encoding, our dataset will look like this.\n'
    writer.write(dialog)
    writer.write('\n' + str(data) + '\n')
    dialog = '\nBefore going further, one thing to note that columns \'case_id\' and \'patientid\' take values in a large range.'
    writer.write(dialog)
    dialog = '\nTherfore, we are removing these columns from the dataset as they have little significance for classification.'
    writer.write(dialog)
    dialog = '\nAlso, the \'Admission_Deposit\' column takes continuous values, for this we divide the amount by 1000 and'
    writer.write(dialog)
    dialog = '\nround it off to nearest integer.\n'
    writer.write(dialog)
    data = data.drop(columns=['case_id', 'patientid'])
    ADM_DEP = data['Admission_Deposit'].copy() # To be used in part 3a
    data['Admission_Deposit'] = data['Admission_Deposit'].apply(lambda x: int(x / 1000 + 0.5))
    dialog = '\nAfter all these pre-processing, our dataset will look like this.\n'
    writer.write(dialog)
    writer.write('\n' + str(data) + '\n')
    writer.write(BREAK_)

    writer.write('Part c\n' + BREAK_)
    df = data.copy()
    df = df.sample(frac=1)
    PARTITION_INDEX = int(PARTITION * df.shape[0])
    train_df = df[:PARTITION_INDEX]
    test_df = df[PARTITION_INDEX:]
    X_train, y_train = getXY(train_df)
    X_test, y_test = getXY(test_df)
    # Performing 5-fold cross validation
    score = KFold(5, train_df, 'plot_1c_cross_val.png')
    writer.write('\nAccuracy of Discrete Naive Bayes after 5-fold cross validation is {0}%\n'.format(score * 100))
    op.write('Part 1c\n' + BREAK_ + 'Accuracy of Discrete Naive Bayes after 5-fold cross validation is {0}%\n'.format(score * 100))
    clf = NaiveBayes()
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test, True, 'plot_1c_final_acc.png')
    writer.write('\nFinal test accuracy of Discrete Naive Bayes is {0}%\n'.format(score * 100))
    op.write('Final test accuracy of Discrete Naive Bayes is {0}%\n'.format(score * 100))

    writer.write('\n\n' + BREAK_ + 'Question 2\n' + BREAK_)

    writer.write('Part a\n' + BREAK_)
    # Let's see how many columns we have
    n_columns = len(list(X_train))
    writer.write('\nNumber of columns in the dataset is {0}\n'.format(n_columns))
    # Using sklearn PCA API, let decompose the dataset onto 'n_columns' principal components
    pca = PCA(n_components=n_columns)
    pca.fit(X_train)
    variances = list(pca.explained_variance_ratio_)
    variances.sort(reverse=True)
    p_comp_names = []
    writer.write('\nVariance explained by principal components:\n')
    writer.write('\nComponents\t\tFraction of Variance Explained\n')
    for i in range(n_columns):
        p_comp_names.append('c{0}'.format(i + 1))
        writer.write('Component_{0}\t\t{1}\n'.format(i + 1, variances[i]))
    plt.switch_backend('Agg')
    plt.xlabel('Components')
    plt.ylabel('Variances')
    plt.bar(p_comp_names, variances)
    plt.title('Scree Graph for PCA components')
    plt.savefig('plot_2a_scree_graph.png')
    plt.close()
    for i in range(1, len(variances)):
        variances[i] += variances[i - 1]
    variances.insert(0, None)
    plt.switch_backend('Agg')
    plt.xlabel('Number of principal components')
    plt.ylabel('Variance preserved')
    plt.plot(variances, label='variance')
    plt.axhline(y=0.95, linewidth=1, color='k')
    plt.title('Proportion of Variance Explained')
    plt.grid(True)
    plt.savefig('plot_2a_variance_proportion_explained.png')
    plt.close()
    # It can be observed from the graph that at least 95% variance is explained
    # if we consider minimum of 7 principal components
    writer.write('\nFor preserving at least 95% of variance, we have to use minimum 7 principal components.\n')
    pca7 = PCA(n_components=7)
    pca7.fit(X_train)
    X_train_pca = pca7.transform(X_train)
    X_test_pca = pca7.transform(X_test)
    writer.write('\nThe training samples after PCA look like this:\n')
    writer.write('\n' + str(X_train_pca) + '\n')
    # After PCA, the dataset is modified from Categorical to Numerical type
    # For such datasets, Discrete Naive Bayes will have a very poor performance
    # Therefore, for this part we introduced Gaussian kernel to account for
    # the continuous valued attributes as NaiveBayesv2, and thus KFoldv2 for
    # carrying out cross-validation

    writer.write(BREAK_ + 'Part b\n' + BREAK_)
    score = KFoldv2(5, train_df, pca7, 'plot_2b_cross_val.png')
    writer.write('\nAccuracy of Gaussian Naive Bayes after 5-fold cross validation is {0}%\n'.format(score * 100))
    op.write('\nPart 2b\n' + BREAK_ + 'Accuracy of Gaussian Naive Bayes after 5-fold cross validation is {0}%\n'.format(score * 100))
    y_train_pca = y_train.to_numpy().squeeze()
    y_test_pca = y_test.to_numpy().squeeze()
    clf = NaiveBayesv2()
    clf.fit(X_train_pca, y_train_pca)
    score = clf.score(X_test_pca, y_test_pca)
    writer.write('\nFinal test accuracy of Gaussian Naive Bayes is {0}%\n'.format(score * 100))
    op.write('Final test accuracy of Gaussian Naive Bayes is {0}%\n'.format(score * 100))

    writer.write('\n\n' + BREAK_ + 'Question 3\n' + BREAK_)

    writer.write('Part a\n' + BREAK_)
    # In our dataset, 'Admission_deposit' is the only column having numerical values
    # other columns have categorical value (and categorical data doesn't have any outliers)
    data_orig = data.copy()
    data_orig['Admission_Deposit'] = ADM_DEP # Decoding 'Admission_Deposit'
    df = data_orig.copy()
    df = df.sample(frac=1)
    # Removing outliers
    df = df[((df['Admission_Deposit'] - df['Admission_Deposit'].mean()) / df['Admission_Deposit'].std()).abs() < 3]
    # Encoding 'Admission_Deposit' again
    df['Admission_Deposit'] = df['Admission_Deposit'].apply(lambda x: int(x / 1000 + 0.5))
    writer.write('\nIn our dataset, \'Admission_Deposit\' is the only column which contained numerical data\n')
    writer.write('which we have encoded in previous step. To detect outliers, we have to decode this column\n')
    writer.write('detect outliers using values in this column, and re-encode the column.\n')
    writer.write('\n# records before removing outliers: {0}\n'.format(len(data_orig)))
    writer.write('\n# records after removing outliers : {0}\n'.format(len(df)))

    writer.write(BREAK_ + 'Part b, c\n' + BREAK_)
    # Selecting best features using Backward Selection Method
    PARTITION_INDEX = int(PARTITION * df.shape[0])
    train_df = df[:PARTITION_INDEX]
    test_df = df[PARTITION_INDEX:]
    X_train, y_train = getXY(train_df)
    X_test, y_test = getXY(test_df)
    best_features = BackwardSelection(X_train, y_train)
    writer.write('\nFeatures present before using backward selection:\n')
    writer.write(str(list(X_train)) + '\n')
    writer.write('\nFeatures selected through Backward selection:\n')
    writer.write(str(best_features) + '\n\n')
    op.write('\nPart 3c\n' + BREAK_)
    op.write('Features selected through Backward selection are:\n')
    op.write(str(best_features) + '\n' + BREAK_)

    writer.write(BREAK_ + 'Part d\n' + BREAK_)
    X_train_trim = X_train[best_features]
    X_test_trim = X_test[best_features]
    best_features = tuple(best_features)
    best_features_withlabel = list(best_features)
    best_features_withlabel.append('Stay')
    train_df_trim = train_df[best_features_withlabel]
    score = KFold(5, train_df_trim, 'plot_3d_cross_val.png')
    writer.write('\nAccuracy of Discrete Naive Bayes after 5-fold cross validation is {0}%\n'.format(score * 100))
    op.write('\nPart 3d\n' + BREAK_ + 'Accuracy of Discrete Naive Bayes after 5-fold cross validation is {0}%\n'.format(score * 100))
    clf = NaiveBayes()
    clf.fit(X_train_trim, y_train)
    score = clf.score(X_test_trim, y_test, True, 'plot_3d_final_acc.png')
    writer.write('\nFinal test accuracy of Discrete Naive Bayes is {0}%\n'.format(score * 100))
    op.write('Final test accuracy of Discrete Naive Bayes is {0}%\n'.format(score * 100))
    writer.close()
    op.close()