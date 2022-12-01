import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.neural_network import MLPClassifier

from enum import Enum
import os


class ClassificationType(Enum):
    SVM_RBF = 1
    BDR = 2
    SVM_LINEAR = 3
    D_TREE = 4
    MLP = 5


def classifier_output(classification_type, training_data, training_labels, test_data):

    classifier_dict = {"SVM_RBF": "perform_svm_rbf", "BDR": "perform_bdr",
                       "SVM_LINEAR": "perform_svm_linear", "D_TREE": "perform_d_tree",
                       "MLP": "perform_mlp_classification"}

    # test_labels = classifier_dict[classification_type.name]()
    test_labels = globals()[classifier_dict[classification_type.name]](training_data, training_labels, test_data)

    return test_labels


def perform_mlp_classification(training_data, training_labels, test_data):
    clf = MLPClassifier()
    clf.fit(training_data, training_labels)
    y_predict = clf.predict(test_data)
    return y_predict


def perform_bdr(training_data, training_labels, test_data):
    gnb = GaussianNB()
    gnb.fit(training_data, training_labels)
    y_pred = gnb.predict(test_data)
    return y_pred


def perform_svm_rbf(training_data, training_labels, test_data):
    clf = svm.SVC(kernel="rbf", decision_function_shape='ovo')
    clf.fit(training_data, training_labels)
    y_predict = clf.predict(test_data)
    return y_predict


def perform_svm_linear(training_data, training_labels, test_data):
    clf = svm.SVC(kernel="linear", decision_function_shape='ovo')
    clf.fit(training_data, training_labels)
    y_predict = clf.predict(test_data)
    return y_predict


def perform_d_tree(training_data, training_labels, test_data):
    clf = tree.DecisionTreeClassifier()
    clf.fit(training_data, training_labels)
    y_pred = clf.predict(test_data)
    return y_pred


def perform_hotel_cluster_classification():
    os.chdir(os.path.realpath(__file__).rsplit("\\", 1)[0])
    orig_dset = pd.read_csv("expedia-hotel-recommendations/train_smallest.csv")

    training_Set_df = orig_dset.dropna()
    training_labels = training_Set_df['hotel_cluster'].to_numpy()

    pure_train = training_Set_df.drop(['date_time', 'cnt', 'is_booking', 'hotel_cluster', 'srch_ci',
                                       'srch_co'], axis=1)
    pure_train_raw = pure_train.to_numpy()
    (training_inp_vec, test_inp_vec, training_set_labels, test_labels) = train_test_split(pure_train_raw, training_labels, test_size=0.2, random_state=999999)

    classification_types_list = ['SVM_RBF', 'BDR', 'D_TREE', 'MLP']

    for classification_type in classification_types_list:
        y_predict_new = classifier_output(getattr(ClassificationType, classification_type), training_inp_vec, training_set_labels, test_inp_vec)
        accuracy_percentage = metrics.accuracy_score(test_labels, y_predict_new)
        print("Method Used - {}".format(classification_type))
        print("Accuracy score %.3f" % accuracy_percentage)

        '''
        cm = confusion_matrix(test_labels, y_predict_new)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(set(training_labels)))
        disp.plot()
        plt.title("Confusion Matrix - {}".format(classification_type))

        plt.show()
        '''


def perform_binary_classification():
    os.chdir(os.path.realpath(__file__).rsplit("\\", 1)[0])
    orig_dset = pd.read_csv("expedia-hotel-recommendations/train_overall.csv")

    training_Set_df = orig_dset.dropna()
    training_labels = training_Set_df['is_booking'].to_numpy()

    pure_train = training_Set_df.drop(['date_time', 'cnt', 'is_booking', 'hotel_cluster', 'srch_ci',
                                       'srch_co'], axis=1)
    pure_train_raw = pure_train.to_numpy()
    (training_inp_vec, test_inp_vec, training_set_labels, test_labels) = train_test_split(pure_train_raw, training_labels, test_size=0.2, random_state=999999)

    classification_types_list = ['SVM_RBF', 'BDR', 'SVM_LINEAR', 'D_TREE']

    for classification_type in classification_types_list:
        y_predict_new = classifier_output(getattr(ClassificationType, classification_type), training_inp_vec, training_set_labels, test_inp_vec)
        accuracy_percentage = metrics.accuracy_score(test_labels, y_predict_new)
        print("Method Used - {}".format(classification_type))
        print("Accuracy score %.3f" % accuracy_percentage)

        cm = confusion_matrix(test_labels, y_predict_new)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
        disp.plot()
        plt.title("Confusion Matrix - {}".format(classification_type))

        plt.show()


def main_function():
    # perform_binary_classification()
    perform_hotel_cluster_classification()


if __name__ == '__main__':
    main_function()

