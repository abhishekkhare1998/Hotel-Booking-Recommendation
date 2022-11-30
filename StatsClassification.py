import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os


def main_function():

    os.chdir(os.path.realpath(__file__).rsplit("\\", 1)[0])

    training_Set_df = pd.read_csv("expedia-hotel-recommendations/train_very_small.csv")
    training_labels = training_Set_df['is_booking'].to_numpy()
    pure_train = training_Set_df.drop(['date_time', 'cnt', 'is_booking', 'hotel_cluster', 'srch_ci',
                                       'srch_co', 'orig_destination_distance'], axis=1)
    pure_train_raw = pure_train.to_numpy()
    (training_inp_vec, test_inp_vec, training_set_labels, test_labels) = train_test_split(pure_train_raw, training_labels, test_size=0.2, random_state=999999)

    clf = svm.SVC(decision_function_shape='ovo')
    clf.fit(training_inp_vec, training_set_labels)

    y_predict = clf.predict(test_inp_vec)
    # Measure the performance
    print("Accuracy score %.3f" % metrics.accuracy_score(test_labels, y_predict))

    cm = confusion_matrix(test_labels, y_predict)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    disp.plot()
    plt.show()


    a = 1

if __name__=='__main__':
    main_function()

