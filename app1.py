import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from datetime import datetime
import networkx as nx
import graphviz

# Set page config
st.set_page_config(page_title="Helpdesk Process Mining App", layout="wide")

# Function to load and preprocess data
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    return df

# Function to create process map using Graphviz
def create_process_map(df, case_id, activity):
    dot = graphviz.Digraph(comment='Process Map')
    dot.attr(rankdir='LR', size='12,8', dpi='300')
    
    # Create nodes
    activities = df[activity].unique()
    for act in activities:
        dot.node(act, act)
    
    # Create edges
    edges = df.groupby(case_id)[activity].apply(lambda x: list(zip(x, x[1:]))).explode()
    edge_counts = edges.value_counts()
    
    for (source, target), count in edge_counts.items():
        dot.edge(source, target, label=str(count), penwidth=str(0.5 + count/edge_counts.max()*2))
    
    return dot

# Function to create transition matrix
def create_transition_matrix(df, case_id, activity):
    df_polars = pl.DataFrame(df)
    transitions = df_polars.groupby(case_id).agg([
        pl.col(activity).slice(1).alias('next_activity'),
        pl.col(activity).alias('current_activity')
    ])
    transition_counts = transitions.select([
        pl.col('current_activity'),
        pl.col('next_activity')
    ]).groupby(['current_activity', 'next_activity']).count()
    
    pivot_table = transition_counts.pivot(
        values='count',
        index='current_activity',
        columns='next_activity',
        aggregate_function='sum'
    ).fill_null(0)
    
    return pivot_table.to_pandas()

# Function to analyze process timing
def analyze_timing(df, case_id, activity, timestamp):
    df['duration'] = df.groupby(case_id)[timestamp].diff().dt.total_seconds() / 3600  # in hours
    timing_stats = df.groupby(activity)['duration'].agg(['mean', 'median', 'max']).reset_index()
    timing_stats = timing_stats.sort_values('mean', ascending=False)
    return timing_stats

# Function to analyze users
def analyze_users(df, resource, activity):
    user_activity = df.groupby([resource, activity]).size().unstack(fill_value=0)
    return user_activity

# Main function
def main():
    st.title("Helpdesk Process Mining App")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        if df is not None:
            st.success("File successfully loaded!")

            # Column selection
            st.sidebar.subheader("Column Selection")
            case_id = st.sidebar.selectbox("Select Case ID column", df.columns)
            activity = st.sidebar.selectbox("Select Activity column", df.columns)
            timestamp = st.sidebar.selectbox("Select Timestamp column", df.columns)
            resource = st.sidebar.selectbox("Select Resource column", df.columns)

            # Process Overview
            st.header("1. Process Overview")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Cases", df[case_id].nunique())
            col2.metric("Total Activities", df[activity].nunique())
            col3.metric("Total Events", len(df))
            col4.metric("Date Range", f"{df[timestamp].min().date()} to {df[timestamp].max().date()}")

            # Process Map
            st.header("2. Process Discovery")
            st.subheader("Process Map")
            process_map = create_process_map(df, case_id, activity)
            st.graphviz_chart(process_map)

            # Transition Matrix
            st.subheader("Transition Matrix")
            transition_matrix = create_transition_matrix(df, case_id, activity)
            fig = px.imshow(transition_matrix, 
                            labels=dict(x="Next Activity", y="Current Activity", color="Frequency"),
                            x=transition_matrix.columns,
                            y=transition_matrix.index)
            st.plotly_chart(fig, use_container_width=True)

            # Timing Analysis
            st.header("3. Timing Analysis")
            timing_stats = analyze_timing(df, case_id, activity, timestamp)
            fig = px.bar(timing_stats, x=activity, y='mean', 
                         error_y='max', error_y_minus='median',
                         labels={'mean': 'Average Duration (hours)', activity: 'Activity'},
                         title="Activity Duration Analysis")
            st.plotly_chart(fig, use_container_width=True)

            # User Analysis
            st.header("4. User Analysis")
            user_activity = analyze_users(df, resource, activity)
            fig = px.imshow(user_activity, 
                            labels=dict(x="Activity", y="User", color="Frequency"),
                            x=user_activity.columns,
                            y=user_activity.index)
            st.plotly_chart(fig, use_container_width=True)

            # Bottleneck Analysis
            st.header("5. Bottleneck Analysis")
            bottlenecks = timing_stats.nlargest(5, 'mean')
            fig = px.bar(bottlenecks, x=activity, y='mean',
                         labels={'mean': 'Average Duration (hours)', activity: 'Activity'},
                         title="Top 5 Bottleneck Activities")
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Please upload a CSV file to begin the analysis.")

if __name__ == "__main__":
    main()
