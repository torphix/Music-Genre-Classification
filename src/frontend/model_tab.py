import numpy as np

OVERVIEW_TEXT = '''
<div style="text-align: center; font-size:20px; width:80%; margin:auto;">
    A mixture of traditional machine learning techniques and neural networks was used to address the task of music genre classification. <br/>
    An initial baseline was established using Kmeans clustering algorithm on the log mel spectrograms extracted from the normalised audio. <br/>
    The low computational overhead and convergance speed made it useful to establish a suitable baseline. <br/>
    However accuracy was low (approx. 15%) to improve upon this baseline one dimensional convolutional neural networks where employed using <br/>
    log mel spectrograms as input features initially and then also further numerical features such as the Chroma stft and RMS values (see data analysis) 
    for an indepth description and visualisation of features. <br/>
    The results are shown below and a simple UI is in place to train the model from scratch (Running on GPU is highly reccomended). <br/>
    Pretrained models are also provided as well as the source code (available under src/models/neural_network). <br/>
    We employed the cannonical resnet architechture https://arxiv.org/abs/1512.03385 Kaiming He et al. replacing the 2D convolutions
    with 1D convolutions. <br/>
    Implementation was done from scratch in pytorch along with a custom learning rate scheduler.
</div>
<br/>  <br/><hr/>
'''



def get_sample_data():
    specs = {
        'blues':np.load('frontend/mel_samples/bluesnpy.npy'),
        'classical':np.load('frontend/mel_samples/classicalnpy.npy'),
        'country':np.load('frontend/mel_samples/countrynpy.npy'),
        'disco':np.load('frontend/mel_samples/disconpy.npy'),
        'hiphop':np.load('frontend/mel_samples/hiphopnpy.npy'),
        'jazz':np.load('frontend/mel_samples/jazznpy.npy'),
        'metal':np.load('frontend/mel_samples/metalnpy.npy'),
        'pop':np.load('frontend/mel_samples/popnpy.npy'),
        'reggae':np.load('frontend/mel_samples/reggaenpy.npy'),
        'rock':np.load('frontend/mel_samples/rocknpy.npy'),
        }
    return specs

