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
