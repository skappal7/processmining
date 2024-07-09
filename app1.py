import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import networkx as nx
import graphviz
from io import StringIO

# Set page config
st.set_page_config(page_title="Advanced Process Mining App", layout="wide")

# Function to load and preprocess data
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    return df

# Function to correct datetime format
def correct_datetime(df, timestamp_col):
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], infer_datetime_format=True)
    return df

# Function to create process map using Graphviz
def create_process_map(df, case_id, activity):
    dot = graphviz.Digraph(engine='dot')
    dot.attr(rankdir='LR')

    # Add nodes
    for act in df[activity].unique():
        dot.node(act, act)

    # Add edges
    edges = df.groupby(case_id)[activity].apply(lambda x: list(zip(x, x[1:]))).explode()
    edge_counts = edges.value_counts()

    for (source, target), count in edge_counts.items():
        dot.edge(source, target, label=str(count))

    return dot

# Function to create transition matrix
def create_transition_matrix(df, case_id, activity):
    transitions = df.groupby(case_id)[activity].apply(lambda x: list(zip(x, x[1:])))
    transition_counts = transitions.explode().value_counts().unstack(fill_value=0)
    return transition_counts

# Main function
def main():
    st.title("Advanced Process Mining App")

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

            # Correct datetime format
            df = correct_datetime(df, timestamp)

            # Process Overview
            st.subheader("Process Overview")
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            col1.metric("Tickets", df[case_id].nunique())
            col2.metric("Activities", df[activity].nunique())
            col3.metric("Events", len(df))
            col4.metric("Users", df['user'].nunique() if 'user' in df.columns else "N/A")
            col5.metric("Products", df['product'].nunique() if 'product' in df.columns else "N/A")
            col6.metric("Variants", df.groupby(case_id)[activity].agg(tuple).nunique())

            # Time range
            st.write(f"Time Range: {df[timestamp].min().date()} to {df[timestamp].max().date()}")

            # Number of cases per month
            st.subheader("Number of cases per month by start date")
            df['month'] = df[timestamp].dt.to_period('M')
            cases_per_month = df.groupby('month')[case_id].nunique().reset_index()
            cases_per_month['month'] = cases_per_month['month'].astype(str)
            fig = px.bar(cases_per_month, x='month', y=case_id, labels={'month': 'Month', case_id: 'Number of Cases'})
            st.plotly_chart(fig, use_container_width=True)

            # Process Map
            st.subheader("Process Map")
            process_map = create_process_map(df, case_id, activity)
            st.graphviz_chart(process_map)

            # Variant Analysis
            st.subheader("Variant Analysis")
            try:
                variants = df.groupby(case_id)[activity].agg(lambda x: tuple(x)).value_counts().reset_index()
                variants.columns = ['Variant', 'Frequency']
                variants['Variant'] = variants['Variant'].astype(str)  # Convert tuple to string for display
                variants = variants.head(10)  # Show top 10 variants

                fig = px.bar(variants, x='Variant', y='Frequency', 
                             labels={'Variant': 'Variant', 'Frequency': 'Frequency'},
                             title='Top 10 Variants')
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)

                # Display variant details in a table
                st.write("Top 10 Variants Details:")
                st.table(variants)
            except Exception as e:
                st.error(f"Error in variant analysis: {str(e)}")
                st.info("This might occur if there are issues with the case ID or activity columns. Please check your data.")

            # Activity Distribution
            st.subheader("Activity Distribution")
            activity_counts = df[activity].value_counts()
            fig = px.bar(activity_counts.reset_index(), x='index', y=activity, labels={'index': 'Activity', activity: 'Frequency'})
            fig.update_layout(xaxis_title="Activity", yaxis_title="Frequency")
            st.plotly_chart(fig, use_container_width=True)

            # Transition Matrix
            st.subheader("Transition Matrix")
            try:
                transition_matrix = create_transition_matrix(df, case_id, activity)
                fig = px.imshow(transition_matrix, labels=dict(x="Next Activity", y="Current Activity", color="Frequency"), 
                                x=transition_matrix.columns, y=transition_matrix.index)
                st.plotly_chart(fig, use_container_width=True)
            except ValueError as e:
                st.warning(f"Unable to create transition matrix. Error: {str(e)}")
                st.info("This might occur if there are cases with only one activity or if activities aren't properly paired.")

    else:
        st.info("Please upload a CSV file to begin the analysis.")

if __name__ == "__main__":
    main()
