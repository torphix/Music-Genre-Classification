import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from src.frontend.utils import feature_desc

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
            chroma_stft_mean, 'chroma stft mean distribution', 0.01, feature_desc.CHROMA_STFT], 
        'rms_mean':[
            rms_mean, 'rms mean, mean of distributions', 0.01, feature_desc.RMS], 
        'mfcc1_mean':[
            mfcc19_mean, 'mfcc1 mean, mean of distributions', 20, feature_desc.RMS], 
    }
