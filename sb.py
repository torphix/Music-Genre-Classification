import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
import seaborn as sns

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
    sns.heatmap(df_cm/5, annot=True)
    plt.show()


# x = plot_confusion_matrix(
# [[3, 0, 1, 0, 0, 1, 0, 0, 0, 0], 
# [0, 5, 0, 0, 0, 0, 0, 0, 0, 0], 
# [2, 0, 1, 1, 0, 0, 0, 0, 0, 1], 
# [0, 0, 1, 2, 0, 0, 0, 2, 0, 0], 
# [0, 0, 0, 0, 4, 0, 0, 0, 1, 0], 
# [0, 0, 0, 0, 0, 5, 0, 0, 0, 0], 
# [0, 0, 0, 0, 1, 0, 4, 0, 0, 0], 
# [0, 0, 0, 0, 1, 0, 0, 4, 0, 0],
#  [0, 0, 0, 0, 0, 0, 0, 1, 3, 1], 
#  [0, 0, 0, 0, 0, 0, 0, 2, 0, 3]]
# )
# print(x)


train_metrics = torch.load('/home/j/Desktop/Programming/Uni/Music-Genre-Classification/logs/50-img-resnet18/train_metrics.pt')
val_metrics = torch.load('/home/j/Desktop/Programming/Uni/Music-Genre-Classification/logs/50-img-resnet18/val_metrics.pt')


plt.plot([i for i in range(len(train_metrics['loss']))], train_metrics['loss'], label='Train Loss')
plt.plot([i for i in range(len(val_metrics['loss']))], val_metrics['loss'], label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc="upper right")
plt.title('Random Initialization Training Loss')
plt.show()
'''
TODO:
1) Display finetune loss and acc curve vs training from scratch + discuss
2) Compare confusion matrices
3) Try with dropout / other norm techniques?
4) Finish frontend 
'''