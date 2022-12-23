import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff

# Functions
def load_data(data_url):
    data = pd.read_csv(data_url)
    return data

def create_barchart(data:list, x_label:str, y_label:str):
    fig = px.bar(data, x=x_label, y=y_label, color=x_label)
    return fig

def create_distchart(data:list, labels:list, bin_size:int):
    fig = ff.create_distplot(data, labels, bin_size=bin_size)
    return fig

def get_feature_distributions(main_df, groups) -> dict:
    '''
    return format is dict of list 
    {"name of feature": [data, distribution_label, bin_size, description]}
    '''
    chroma_stft_mean = [main_df.iloc[groups.groups[k]]['chroma_stft_mean'].to_list() 
                             for k,v in groups.groups.items()]
    rms_mean = [main_df.iloc[groups.groups[k]]['rms_mean'].to_list() 
                for k,v in groups.groups.items()]
    mfcc19_mean = [main_df.iloc[groups.groups[k]]['mfcc1_mean'].to_list() 
                for k,v in groups.groups.items()]
    return {
        'chroma_stft_mean':[
            chroma_stft_mean, 'chroma stft mean distribution', 0.01, CHROMA_STFT], 
        'rms_mean':[
            rms_mean, 'rms mean, mean of distributions', 0.01, RMS], 
        'mfcc1_mean':[
            mfcc19_mean, 'mfcc1 mean, mean of distributions', 20, MFCC], 
    }



CHROMA_STFT = '''
<div style="text-align: center; font-size:20px">
    Chroma short time fourier transform describes the harmonic and melodic characteristics of audio whilst being sturdy against changes in instrumentation.
</div>
<div style="text-align: center; font-size:20px">
    The distributions of the extracted chroma stft's are approximately normally distributed with shifts in the mean and variance depending on the musical genre.
</div>
<div style="text-align: center; font-size:20px">
    This makes it possible to predict a likelihood of which type of musical genre a chroma stft is from based upon its value.
</div>
<br/>  <br/><hr/>
'''
RMS = '''
<div style="text-align: center; font-size:20px">
    Root mean square (RMS) measures average loudness of the music
</div>
<div style="text-align: center; font-size:20px">
    Different music styles are likley to have different intensities of volume, however RMS may also be a confounding variable
</div>
<div style="text-align: center; font-size:20px">
    as environmental conditions may have an impact such as distance of instrument from the microphone, recording equipment used and audio postprocessing techniques.
</div>
<div style="text-align: center; font-size:20px; ">
    As seen in the data distributions there is signal in the RMS value in particular classical music is generally significantly quieter ie: smaller RMS values than 
    the other genres as even loud classical music generally has several expressive crescendos and diminuendos throughout.
</div>
<br/>  <br/><hr/>
'''
MFCC = '''
<div style="text-align: center; font-size:20px; width:80%; margin:auto;">
    Mel-frequency cepstral coefficients is a way of extracting features from audio. <br/>
    And are commonly used in deep learning audio applications <br/>
    A brief description of the algorithm is given below: <br/>
    1) Take a window of the audio and apply the fourier transform decomposing the signal into its components <br/>
    2) Map the spectrum into the mel scale <br/>
    3) Normalise by taking the log of the powers of each mel frequency <br/>
    4) Take the discrete cosine transform <br/>
    5) Take the amplitudes of the resulting spectrum <br/>
    As can be seen in the plot above whilst the distributions are of a different shape, there are notable differences making 
    MFCC's a useful feature to include in model training.
</div>
<br/>  <br/><hr/>
'''
