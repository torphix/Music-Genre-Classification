import json
import pickle
import torch
import pathlib
import pandas as pd
import librosa.display
import streamlit as st
import plotly.express as ptx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from data_tab import (
    load_data, 
    get_feature_distributions, 
    create_distchart, 
    create_barchart)
from model_tab import (
    OVERVIEW_TEXT,
    get_sample_data,
)
# Constants
ROOT = pathlib.Path(__file__).parent.parent.parent
DATA_URL = f'{ROOT}/data/features_30_sec.csv'

# App UI
def main_ui():
    st.set_page_config(layout="wide")
    st.markdown('''
    <div style="text-align: center; font-size:50px">
        <b>Music Genre Analysis<b/>
    </div>
    ''', unsafe_allow_html=True)

    overview_tab, data_analysis_tab, models_tab = st.tabs(["Overview", "Data Analysis", "Models"])

    with overview_tab:  
        overview_tab_ui()
    
    with data_analysis_tab:
        data_analysis_tab_ui(DATA_URL)
    
    with models_tab:
        models_tab_ui()    
   


def overview_tab_ui():
    st.markdown('''
    <div style="text-align: center; font-size:40px; width:80%; margin:auto;">
        <b>Project Overview</b>
    </div>
    ''', unsafe_allow_html=True)
    st.markdown(OVERVIEW_DESCRIPTION, unsafe_allow_html=True)
    st.markdown('''
    <div style="text-align: center; font-size:40px; width:80%; margin:auto;">
        <b>Links & References</b>
    </div>
    ''', unsafe_allow_html=True)
    st.markdown(LINKS_AND_REFERENCES, unsafe_allow_html=True)

    

def data_analysis_tab_ui(data_url):
     # Create a text element and let the reader know the data is loading.
    # Load 10,000 rows of data into the dataframe.
    data = load_data(data_url)
    # Notify the reader that the data was successfully loaded.
    st.subheader(f'Raw data, Length: {len(data)}')
    st.write(data)
    st.text('Feature Statistics')
    # Data Analysis
    # Split data by genre category
    groups = data.groupby('label', as_index=False)
    # Plot histograms of the mean of each feature for each group
    avg_values = groups.mean()
    group_labels = list(groups.groups.keys())
    feature_distributions = get_feature_distributions(data, groups)
    
    for k, (dist, header, bin_size, desc) in feature_distributions.items():
        col1, col2 = st.columns([4,4])
        with col1:    
            col1.header(header)
            st.plotly_chart(create_distchart(dist, group_labels, bin_size), use_container_width=True)
        with col2:
            col2.header(f'{" ".join(k.split("_"))}, mean of distributions')
            st.plotly_chart(create_barchart(avg_values, 'label', k))
        st.markdown(desc, unsafe_allow_html=True)

    st.markdown('''
                <div style="text-align: center; font-size:40px; width:80%; margin:auto;">
                    <b>Mel Spectrograms, Neural Network Input Data</b> <br/>
                    <p>If some images are not loading click the models tab then return to this tab<p/>
                </div>''',
                    unsafe_allow_html=True)
    sample_mels = get_sample_data()
    # Row 1
    cols = st.columns(5)
    data = list(iter(sample_mels.items()))
    for i in range(len(cols)):
        with cols[i]:
            st.markdown(f'<div style="text-align: center; font-size:25px; width:80%; margin:auto;"><b>{data[i][0]}</b></div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(10,10))
            img = librosa.display.specshow(data[i][1], ax=ax)
            fig.colorbar(img, ax=ax)
            st.pyplot(fig)
            st.markdown(f'<div style="text-align: center; font-size:25px; width:80%; margin:auto;"><b>{data[i+5][0]}</b></div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(10,10))
            img = librosa.display.specshow(data[i+5][1], ax=ax)
            fig.colorbar(img, ax=ax)
            st.pyplot(fig)
    st.markdown('<hr/>', unsafe_allow_html=True)
    

def models_tab_ui():
    # Overview
    st.markdown('<div style="text-align: center; font-size:40px; width:80%; margin:auto;"><b>Summary of Approach</b></div>', unsafe_allow_html=True)
    st.markdown(OVERVIEW_TEXT, unsafe_allow_html=True)
    st.markdown('<div style="text-align: center; font-size:40px; width:80%; margin:auto;"><b>Training Summary of Best Model</b></div>', unsafe_allow_html=True)
    train_losses = torch.load('../logs/highest_score/train_metrics.pt', 'cpu')
    val_losses = torch.load('../logs/highest_score/val_metrics.pt', 'cpu')

    with open('../logs/highest_score/config.json', 'r') as f:
        model_config = json.loads(f.read())

    # Losses
    st.title('Training Losses')
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=[i for i in range(len(train_losses['loss']))], 
                   y=train_losses['loss'],
                   mode='lines+markers',
                   name='Train Loss'))
    fig.add_trace(
        go.Scatter(x=[i for i in range(len(val_losses['loss']))], 
                   y=val_losses['loss'],
                   mode='lines+markers',
                   name='Validation Loss'))
    st.plotly_chart(fig, use_container_width=True, title='Loss')

    # Hyperparamters
    st.title('Model Hyperparameters')
    st.table(pd.DataFrame([
        ['Architecture',model_config['architechture']],
        ['Parameters', model_config['model_parameters']],
        ['Number of Epochs', model_config['epochs']],
        ['Batch Size', model_config['batch_size']],
        ], index=None))
    
    # Results
    st.title('Neural Network Results')
    col1, col2 = st.columns([4,4])
    with col1:    
        labels = ['blues', 'classical', 'country', 'disco','hiphop','jazz','metal','pop','reggae','rock']
        df_cm = pd.DataFrame(model_config['testset_metrics']['cf_matrix'], 
                            index = labels,
                            columns = labels)
        fig = ptx.imshow(df_cm/50, text_auto=True, title='Confusion Matrix (Percentages)')
        st.plotly_chart(fig)
    with col2:
        st.text('Metrics')
        st.image('../logs/highest_score/metrics.png', width=500)
        

    st.title('Machine Learning Results')
    col1, col2 = st.columns([4,4])
    with col1:
        with open('../logs/highest_score/svm_cm.pickle', 'rb') as f:
            cm = pickle.load(f)
        fig = ptx.imshow(cm, text_auto=True, title='SVM confusion Matrix (3-sec aggregated)')
        st.plotly_chart(fig)

    with col2:
        with open('../logs/highest_score/ml_techniques_df.pickle', 'rb') as f:
            df = pickle.load(f)

        fig = ptx.bar(df, x='model', y='accuracy', title='Summary of Classical ML Techniques used as baseline')
        st.plotly_chart(fig)


OVERVIEW_DESCRIPTION = '''
<div style="text-align: center; font-size:20px; width:80%; margin:auto;">
    Given a dataset of 30 second music clips and various numerical features extracted from the data we seek to classify 
    the music clips into the respective musical genre. <br/>
    To commence we perform an intial cursory data analysis exploring the best numerical feature types that can be extracted from the audio clips. <br/>
    Following several data cleaning techniques, an in depth disucssion into the cleaning, normalisation and features are provided under the data tab. <br/>
    We employ classical machine learning techniques such as KNN, linear regression and  Kmeans clustering to establish a suitable baseline. <br/>
    We iteratively improve upon the established baseline using various neural network architechtures such as <a href="https://arxiv.org/abs/1512.03385">resnets Kaimin He et al.</a> <br/>
    Our various experiments with different architechtures and the impact on selected metrics are discussed and analysed under the models tab. <br/>
</div>
'''
LINKS_AND_REFERENCES = '''
<div style="text-align: center; font-size:20px; width:80%; margin:auto;">
    <b>Github Link:</b> https://github.com/torphix/Music-Genre-Classification<br/>
    <b>Group Members:</b> Jesse Deng, Elisa Troschka <br/>
    <b>Dataset source link:</b> https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification
</div>
'''


if __name__ == '__main__':
    
    main_ui()
