import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import networkx as nx
import matplotlib.pyplot as plt
from io import StringIO, BytesIO

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

# Function to generate process model
def generate_process_model(df):
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['end_time'] = pd.to_datetime(df['end_time'])
    df.sort_values(by=['start_time'], inplace=True)
    events = df[['case_id', 'activity', 'start_time']].values
    G = nx.DiGraph()
    for i in range(len(events) - 1):
        if events[i][0] == events[i + 1][0]:  # same case_id
            G.add_edge(events[i][1], events[i + 1][1])
    return G

# Function to display process model
def display_process_model(G):
    pos = nx.spring_layout(G)
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color="skyblue", font_size=15, font_weight="bold")
    st.pyplot(plt)

# Function for performance analysis
def performance_analysis(df):
    df['duration'] = (df['end_time'] - df['start_time']).dt.total_seconds()
    performance_summary = df.groupby('activity')['duration'].describe()
    st.subheader("Performance Analysis")
    st.write(performance_summary)

# Function for conformance checking
def conformance_checking(df, expected_model):
    actual_model = generate_process_model(df)
    differences = list(set(expected_model.edges()) - set(actual_model.edges()))
    st.subheader("Conformance Checking")
    st.write("Differences between expected and actual process models:")
    st.write(differences)

# Function for bottleneck identification
def bottleneck_identification(df):
    df['duration'] = (df['end_time'] - df['start_time']).dt.total_seconds()
    bottlenecks = df.groupby('activity')['duration'].mean().sort_values(ascending=False)
    st.subheader("Bottleneck Identification")
    st.write(bottlenecks)

# Function for root cause analysis
def root_cause_analysis(df):
    df['duration'] = (df['end_time'] - df['start_time']).dt.total_seconds()
    causes = df.groupby('case_id')['duration'].sum().sort_values(ascending=False)
    st.subheader("Root Cause Analysis")
    st.write(causes)

# Function for variant analysis
def variant_analysis(df):
    df['path'] = df.groupby('case_id')['activity'].transform(lambda x: ' -> '.join(x))
    variants = df['path'].value_counts()
    st.subheader("Variant Analysis")
    st.write(variants)

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

        # Generate and display process model
        st.subheader("Process Model")
        G = generate_process_model(data)
        display_process_model(G)

        # Performance Analysis
        performance_analysis(data)

        # Conformance Checking
        # Define an expected model based on actual activities
        expected_model = nx.DiGraph()
        expected_model.add_edges_from([
            ("Call Received", "Call Transferred"),
            ("Call Transferred", "Call Handled"),
            ("Call Handled", "Information Provided"),
            ("Information Provided", "Call Ended"),
        ])
        conformance_checking(data, expected_model)

        # Bottleneck Identification
        bottleneck_identification(data)

        # Root Cause Analysis
        root_cause_analysis(data)

        # Variant Analysis
        variant_analysis(data)
