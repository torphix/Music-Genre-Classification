import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
import seaborn as sns

def datatype_accuracy_plot():
    testset_acc = [39, 73, 72, 76]
    objects = ['audio', 'img', 'mel', 'multimodal']
    y_pos = np.arange(len(testset_acc))


    plt.bar(y_pos, testset_acc, align='center', alpha=0.5)
    plt.ylim(0, 100)
    plt.xticks(y_pos, objects)
    plt.ylabel('Testset accuracy')
    plt.title('Effect of Input Data type on Accuracy')
    plt.show()


def model_size_difference_acc_plot():
    testset_acc = [65, 75.0, 74, 73]
    objects = ['resnet18', 'resnet50', 'resnet101', 'resnet152']
    y_pos = np.arange(len(testset_acc))

    plt.bar(y_pos, testset_acc, align='center', alpha=0.5)
    plt.ylim(0, 100)
    plt.xticks(y_pos, objects)
    plt.ylabel('Testset accuracy')
    plt.title('Effect of Model Size on Accuracy')
    plt.show()


def plot_confusion_matrix(cf_matrix):
    df_cm = pd.DataFrame(cf_matrix, 
                        index = [i for i in "ABCDEFGHIJK"],
                        columns = [i for i in "ABCDEFGHIJK"])
    plt.figure(figsize = (10,7))
    sns.heatmap(df_cm, annot=True)



x = plot_confusion_matrix(
    [[7, 0, 0, 0, 0, 1, 0, 0, 1, 0], [0, 7, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 2, 0, 0, 0, 0, 0, 0, 1], [0, 0, 2, 3, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 1, 0, 0], [1, 0, 0, 2, 0, 2, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 5, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 5, 0, 1], [1, 0, 1, 1, 0, 0, 0, 0, 3, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]
)
# print(x)