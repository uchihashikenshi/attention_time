#coding:utf-8
import numpy
import matplotlib.pyplot as plt

from sklearn.grid_search import GridSearchCV
import sklearn

class Visualize():

    def __init__(self):
        pass

    def precision_recall_curve(self, label, decision_score, name):

        precision, recall, _ = sklearn.metrics.precision_recall_curve(label, decision_score)
        average_precision = sklearn.metrics.average_precision_score(label, decision_score)

        plt.clf()
        plt.plot(recall, precision, label = 'Precision-Recall curve')
        plt.xlabel('Recall')
        plt.ylabel('precision')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.title('%s: average_precision = {0:0.2f}'.format(average_precision) % name)
        plt.legend(loc='lower left')
        plt.show()


    def roc_curve(self, learner_name, test, test_score):

        fpr, tpr, thresholds = sklearn.metrics.roc_curve(test['y'], test_score)
        roc_auc = sklearn.metrics.auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, label='ROC curve (area = %0.5f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('%s' % learner_name)
        plt.legend(loc="lower right")
        plt.show()


