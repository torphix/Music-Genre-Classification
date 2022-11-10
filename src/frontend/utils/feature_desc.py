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
