import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from datetime import datetime
import io
import graphviz

# Set page config
st.set_page_config(page_title="Comprehensive Process Mining App", layout="wide")

# Function to load and preprocess data
def load_data(file):
    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
    elif file.name.endswith('.xes'):
        # You might need to implement XES parsing here
        st.error("XES parsing not implemented yet.")
        return None
    else:
        st.error("Unsupported file format. Please upload a CSV or XES file.")
        return None
    return df

# Function to correct datetime format
def correct_datetime(df, timestamp_col):
    try:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], infer_datetime_format=True)
    except:
        st.error("Unable to parse timestamp column. Please ensure it contains valid dates.")
        return None
    return df

# Function to create process map
def create_process_map(df, case_id, activity):
    dot = graphviz.Digraph()
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
    st.title("Comprehensive Process Mining App")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        if df is not None:
            st.success("File successfully loaded!")

            # Column selection
            st.subheader("Column Selection")
            case_id = st.selectbox("Select Case ID column", df.columns)
            activity = st.selectbox("Select Activity column", df.columns)
            timestamp = st.selectbox("Select Timestamp column", df.columns)

            # Correct datetime format
            df = correct_datetime(df, timestamp)
            if df is None:
                return

            # Process Overview
            st.subheader("Process Overview")
            total_cases = df[case_id].nunique()
            total_events = len(df)
            unique_activities = df[activity].nunique()
            st.write(f"Total Cases: {total_cases}")
            st.write(f"Total Events: {total_events}")
            st.write(f"Unique Activities: {unique_activities}")

            # Variant Analysis
            st.subheader("Variant Analysis")
            variants = df.groupby(case_id)[activity].agg(lambda x: tuple(x)).value_counts()
            st.write("Top 5 Variants:")
            st.write(variants.head())

            # Time Analysis
            st.subheader("Time Analysis")
            df['duration'] = df.groupby(case_id)[timestamp].diff().dt.total_seconds() / 86400  # Convert to days
            mean_duration = df.groupby(case_id)['duration'].sum().mean()
            median_duration = df.groupby(case_id)['duration'].sum().median()
            st.write(f"Mean Process Duration: {mean_duration:.2f} days")
            st.write(f"Median Process Duration: {median_duration:.2f} days")

            # Visualizations
            st.subheader("Visualizations")

            # Process Map
            st.write("Process Map")
            process_map = create_process_map(df, case_id, activity)
            st.graphviz_chart(process_map)

            # Activity Distribution
            st.write("Activity Distribution")
            fig, ax = plt.subplots(figsize=(12, 6))
            df[activity].value_counts().plot(kind='bar', ax=ax)
            plt.title("Activity Distribution")
            plt.xlabel("Activity")
            plt.ylabel("Frequency")
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)

            # Case Duration Distribution
            st.write("Case Duration Distribution")
            case_durations = df.groupby(case_id)[timestamp].agg(['min', 'max']).diff(axis=1)['max'].dt.total_seconds() / 86400
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.histplot(case_durations, kde=True, ax=ax)
            plt.title("Case Duration Distribution")
            plt.xlabel("Duration (days)")
            st.pyplot(fig)

            # Transition Matrix
            st.write("Transition Matrix")
            transition_matrix = create_transition_matrix(df, case_id, activity)
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(transition_matrix, annot=True, cmap="YlGnBu", ax=ax)
            plt.title("Activity Transition Matrix")
            st.pyplot(fig)

            # First and Last Activities
            st.write("First and Last Activities")
            first_activities = df.groupby(case_id)[activity].first().value_counts()
            last_activities = df.groupby(case_id)[activity].last().value_counts()
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            first_activities.plot(kind='bar', ax=ax1, title="First Activities")
            last_activities.plot(kind='bar', ax=ax2, title="Last Activities")
            ax1.tick_params(axis='x', rotation=45)
            ax2.tick_params(axis='x', rotation=45)
            st.pyplot(fig)

            # Events per Case
            st.write("Events per Case")
            events_per_case = df.groupby(case_id).size()
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.histplot(events_per_case, kde=True, ax=ax)
            plt.title("Events per Case Distribution")
            plt.xlabel("Number of Events")
            st.pyplot(fig)

    else:
        st.info("Please upload a CSV file to begin the analysis.")

if __name__ == "__main__":
    main()
