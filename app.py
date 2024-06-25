import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pm4py
from pm4py.visualization.petrinet import visualizer as pn_visualizer
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.util import dataframe_utils
from io import BytesIO
import matplotlib.pyplot as plt

st.set_page_config(page_title="Process Mining App", layout="wide")

# Function to load data
def load_data(uploaded_file):
    if uploaded_file.name.endswith('.csv'):
        return pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        return pd.read_excel(uploaded_file)
    else:
        st.error("Unsupported file type. Please upload a CSV or Excel file.")
        return None

# Function to visualize the Petri net
def visualize_petri_net(net, initial_marking, final_marking):
    gviz = pn_visualizer.apply(net, initial_marking, final_marking)
    return pn_visualizer.view(gviz)

# Main App
st.title("Process Mining App for Call Centers")

st.sidebar.header("Upload Data")
st.sidebar.write("Please upload a CSV or Excel file with the following columns:")
st.sidebar.write("- case_id: Unique identifier for each call/process instance")
st.sidebar.write("- activity: Name of the activity or event")
st.sidebar.write("- start_time: Start time of the activity (format: YYYY-MM-DD HH:MM:SS)")
st.sidebar.write("- end_time: End time of the activity (format: YYYY-MM-DD HH:MM:SS)")

uploaded_file = st.sidebar.file_uploader("Choose a CSV or Excel file", type=['csv', 'xlsx'])

if uploaded_file:
    data = load_data(uploaded_file)
    if data is not None:
        st.subheader("Uploaded Data")
        st.write(data.head())
        
        # Convert the data into an event log
        data = dataframe_utils.convert_timestamp_columns_in_df(data)
        log = log_converter.apply(data)

        # Discover the process model using the Inductive Miner
        net, initial_marking, final_marking = inductive_miner.apply(log)
        
        # Visualize the process model
        st.subheader("Discovered Process Model (Petri Net)")
        st.pyplot(visualize_petri_net(net, initial_marking, final_marking))
        
        # Conformance Checking using Token-based replay
        st.subheader("Conformance Checking")
        replayed_traces = token_replay.apply(log, net, initial_marking, final_marking)
        st.write("Conformance checking results:")
        st.write(replayed_traces)

        # Performance Analysis
        st.subheader("Performance Analysis")
        performance_df = pm4py.statistics.traces.generic.log_statistics.get_all_trace_variants(log)
        performance_df = pd.DataFrame.from_dict(performance_df, orient='index').reset_index()
        performance_df.columns = ['Trace Variant', 'Count']
        st.write("Trace variants and their counts:")
        st.write(performance_df)

        # Bottleneck Identification
        st.subheader("Bottleneck Identification")
        bottleneck_df = pm4py.statistics.sojourn_time.log.get_sojourn_time(log)
        st.write("Sojourn time (time spent) in each activity:")
        st.write(bottleneck_df)

        # Root Cause Analysis (Simple version: identifying longest running cases)
        st.subheader("Root Cause Analysis")
        root_cause_df = data.groupby('case_id').apply(lambda x: (x['end_time'] - x['start_time']).sum()).reset_index()
        root_cause_df.columns = ['case_id', 'total_duration']
        root_cause_df = root_cause_df.sort_values(by='total_duration', ascending=False)
        st.write("Cases with the longest duration:")
        st.write(root_cause_df.head(10))

        # Variant Analysis
        st.subheader("Variant Analysis")
        variants_df = pm4py.statistics.traces.generic.log_statistics.get_variant_statistics(log)
        variants_df = pd.DataFrame(variants_df)
        st.write("Variants and their frequencies:")
        st.write(variants_df[['variant', 'count']])

