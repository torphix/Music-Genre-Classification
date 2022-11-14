import os
import torch
import numpy as np
import pandas as pd
import librosa.display
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from src.frontend.utils.data_tab import (
    load_data, 
    get_feature_distributions, 
    create_distchart, 
    create_barchart)
from src.frontend.utils.model_tab import (
    OVERVIEW_TEXT,
    extract_features,
    get_sample_data,
)
from src.frontend.utils.overview_tab import (
    OVERVIEW_DESCRIPTION,
    LINKS_AND_REFERENCES,
)
from src.models.neural_network.train import Trainer


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

        # Data preprocessing 
    if os.path.exists('data/mel_samples') == False:
        st.markdown('<div style="text-align: center; font-size:40px; width:80%; margin:auto;"><b>Preprocess Input Data</b></div>', unsafe_allow_html=True)
        _, center, _ = st.columns(3)
        with center:
            data_processor_bar = st.progress(0)
            st.button('Extract input features', on_click=extract_features, args=([data_processor_bar]))
    else:
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
    st.markdown('<div style="text-align: center; font-size:40px; width:80%; margin:auto;"><b>Training Summary</b></div>', unsafe_allow_html=True)
    losses = torch.load('logs/60.losses', 'cpu')

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(losses['train_losses'].keys()), 
                            y=list(losses['train_losses'].values()),
                            mode='lines+markers',
                            name='Train Loss'))
    fig.add_trace(go.Scatter(x=list(losses['val_losses'].keys()), 
                            y=list(losses['val_losses'].values()),
                            mode='lines+markers',
                            name='Validation Loss'))
    st.plotly_chart(fig, use_container_width=True)
    # trainer = Trainer()

