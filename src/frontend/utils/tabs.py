import streamlit as st
from src.frontend.utils.data_tab import (
    load_data, 
    get_feature_distributions, 
    create_distchart, 
    create_barchart)
from src.frontend.utils.model_tab import (
    OVERVIEW_TEXT
)

def overview_tab_ui():
    pass

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


def models_tab_ui():
    # Overview
    st.markdown('<div style="text-align: center; font-size:40px; width:80%; margin:auto;"><b>Summary of Approach</b></div>', unsafe_allow_html=True)
    st.markdown(OVERVIEW_TEXT, unsafe_allow_html=True)
    # Data preprocessing 
    st.markdown('<div style="text-align: center; font-size:40px; width:80%; margin:auto;"><b>Preprocess Input Data</b></div>', unsafe_allow_html=True)
