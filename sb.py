import torch
import plotly.express as py

x = torch.load('/home/j/Desktop/Programming/Uni/Music-Genre-Classification/logs/100-img/train_metrics.pt')
val = torch.load('/home/j/Desktop/Programming/Uni/Music-Genre-Classification/logs/100-img/val_metrics.pt')
# x['loss'] = [i.detach().cpu() for i in x['loss']]
# val['loss'] = [i.detach().cpu() for i in val['loss']]
# fig = py.line(x=[i for i in range(len(x['loss']))], y=[x['loss'], val['loss']])


x['acc'] = [i.detach().cpu() for i in x['acc']]
val['acc'] = [i.detach().cpu() for i in val['acc']]
# fig = py.line(x=[i for i in range(len(x['acc']))], y=[x['acc'], val['acc']], title=)
fig.show()
