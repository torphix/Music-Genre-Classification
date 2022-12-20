import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt


def plot_losses(losses):
    fig = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])
    fig.show()



def datatype_accuracy_plot():
    testset_acc = [43, 60, 57, 71]
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
    labels = ['blues', 'classical', 'country', 'disco','hiphop','jazz','metal','pop','reggae','rock']
    df_cm = pd.DataFrame(cf_matrix, 
                         index = labels,
                         columns = labels)
    plt.figure(figsize = (10,7))
    sns.heatmap(df_cm/50, annot=True)
    plt.show()