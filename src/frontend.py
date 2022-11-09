import pathlib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.figure_factory as ff


# Constants
ROOT = pathlib.Path(__file__).parent.parent
DATA_URL = f'{ROOT}/data/features_30_sec.csv'

# Functions
def load_data():
    data = pd.read_csv(DATA_URL)
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
    {"name of feature": [data, distribution_label, bin_size]}
    '''
    chroma_stft_mean = [main_df.iloc[groups.groups[k]]['chroma_stft_mean'].to_list() 
                             for k,v in groups.groups.items()]
    rms_mean = [main_df.iloc[groups.groups[k]]['rms_mean'].to_list() 
                for k,v in groups.groups.items()]
    return {
        'chroma_stft_mean':[chroma_stft_mean, 'chroma stft mean distribution (Average loudness)', 0.01], 
        'rms_mean':[rms_mean, 'rms mean, mean of distributions', 0.01], 
    }

# App UI
def main_ui():
    st.set_page_config(layout="wide")
    st.title('Music Genre Analysis')
    # Create a text element and let the reader know the data is loading.
    data_load_state = st.text('Loading data...')
    # Load 10,000 rows of data into the dataframe.
    data = load_data()
    # Notify the reader that the data was successfully loaded.
    data_load_state.text('Loading data...done!')
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
    
    for k, (dist, header, bin_size) in feature_distributions.items():
        col1, col2 = st.columns([4,4])
        with col1:    
            col1.write(header)
            st.plotly_chart(create_distchart(dist, group_labels, bin_size), use_container_width=True)
        with col2:
            col2.write(f'{" ".join(k.split("_"))}, mean of distributions')
            st.plotly_chart(create_barchart(avg_values, 'label', k))

if __name__ == '__main__':
    main_ui()
