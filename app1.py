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
def create_process_map(df, case_id, activity, resource=None):
    dot = graphviz.Digraph(comment='Process Map')
    dot.attr(rankdir='LR', 
             ranksep='0.5',
             nodesep='0.5',
             pad='0.5',
             compound='true',
             concentrate='true',
             splines='ortho')
    
    try:
        # Create a subgraph for start activities
        with dot.subgraph(name='cluster_start') as c:
            c.attr(label='Start Activities', style='filled', color='lightgrey')
            start_activities = df.groupby(case_id)[activity].first().value_counts()
            for act, count in start_activities.items():
                c.node(str(act), f"{act}\n({count})", shape='box', style='filled', fillcolor='lightblue')

        # Create a subgraph for end activities
        with dot.subgraph(name='cluster_end') as c:
            c.attr(label='End Activities', style='filled', color='lightgrey')
            end_activities = df.groupby(case_id)[activity].last().value_counts()
            for act, count in end_activities.items():
                c.node(str(act), f"{act}\n({count})", shape='box', style='filled', fillcolor='lightpink')

        # Add other activities
        other_activities = set(df[activity].unique()) - set(start_activities.index) - set(end_activities.index)
        for act in other_activities:
            count = df[df[activity] == act].shape[0]
            dot.node(str(act), f"{act}\n({count})", shape='box')

        # Add edges
        if resource:
            edges = df.groupby(case_id).apply(lambda x: list(zip(x[activity], x[activity].shift(-1), x[resource]))).explode()
            edge_counts = edges.value_counts()
            for (source, target, res), count in edge_counts.items():
                if pd.notna(target):
                    dot.edge(str(source), str(target), label=f"{res}\n({count})", penwidth=str(0.5 + count/edge_counts.max()*2))
        else:
            edges = df.groupby(case_id)[activity].apply(lambda x: list(zip(x, x[1:]))).explode()
            edge_counts = edges.value_counts()
            for (source, target), count in edge_counts.items():
                dot.edge(str(source), str(target), label=str(count), penwidth=str(0.5 + count/edge_counts.max()*2))

        return dot
    except Exception as e:
        raise ValueError(f"Error in creating process map: {str(e)}")

# Function to create variant flow
def create_variant_flow(df, case_id, activity, top_n=20):
    variants = df.groupby(case_id)[activity].agg(list).value_counts()
    top_variants = variants.head(top_n)

    variant_data = []
    for i, (variant, count) in enumerate(top_variants.items(), 1):
        for j, act in enumerate(variant):
            variant_data.append({
                'Variant': f'Variant {i}',
                'Activity': act,
                'Position': j,
                'Count': count
            })

    df_plot = pd.DataFrame(variant_data)
    unique_activities = df[activity].unique()
    color_map = {act: f'hsl({i*360/len(unique_activities)},50%,60%)' for i, act in enumerate(unique_activities)}

    fig = go.Figure()

    for variant in df_plot['Variant'].unique():
        variant_df = df_plot[df_plot['Variant'] == variant]
        fig.add_trace(go.Bar(
            y=[variant] * len(variant_df),
            x=variant_df['Position'],
            customdata=variant_df[['Activity', 'Count']],
            orientation='h',
            marker=dict(color=[color_map[act] for act in variant_df['Activity']]),
            hovertemplate='Activity: %{customdata[0]}<br>Count: %{customdata[1]}<extra></extra>',
            name=variant
        ))

    fig.update_layout(
        title='Top Variants Flow',
        barmode='stack',
        yaxis=dict(title='Variant', categoryorder='total ascending'),
        xaxis=dict(title='Activity Sequence', dtick=1),
        showlegend=False,
        height=800
    )

    return fig

# Function to create transition matrix
def create_transition_matrix(df, case_id, activity):
    transitions = df.groupby(case_id)[activity].apply(lambda x: list(zip(x, x[1:])))
    transition_counts = transitions.explode().value_counts().unstack(fill_value=0)
    return transition_counts

# Function to add explanation
def add_explanation(title, explanation):
    with st.expander(f"ℹ️ {title} Explanation"):
        st.write(explanation)

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
            resource = st.sidebar.selectbox("Select Resource column (optional)", ['None'] + list(df.columns))
            resource = None if resource == 'None' else resource

            # Correct datetime format
            df = correct_datetime(df, timestamp)

            # Process Overview
            st.subheader("Process Overview")
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            col1.metric("Tickets", df[case_id].nunique())
            col2.metric("Activities", df[activity].nunique())
            col3.metric("Events", len(df))
            col4.metric("Users", df[resource].nunique() if resource else "N/A")
            col5.metric("Products", df['product'].nunique() if 'product' in df.columns else "N/A")
            col6.metric("Variants", df.groupby(case_id)[activity].agg(tuple).nunique())
            
            add_explanation("Process Overview", "This section provides key metrics about the process. Tickets represent unique cases, Activities are distinct process steps, Events are total occurrences of activities, Users are distinct resources (if applicable), Products are distinct product types (if applicable), and Variants are unique sequences of activities.")

            # Time range
            st.write(f"Time Range: {df[timestamp].min().date()} to {df[timestamp].max().date()}")

            # Number of cases per month
            st.subheader("Number of cases per month by start date")
            df['month'] = df[timestamp].dt.to_period('M')
            cases_per_month = df.groupby('month')[case_id].nunique().reset_index()
            cases_per_month['month'] = cases_per_month['month'].astype(str)
            fig = px.bar(cases_per_month, x='month', y=case_id, labels={'month': 'Month', case_id: 'Number of Cases'})
            st.plotly_chart(fig, use_container_width=True)
            
            add_explanation("Cases per Month", "This chart shows the distribution of cases over time. It helps identify trends, seasonality, or unusual spikes in process instances.")

            # Process Map
            st.subheader("Process Map")
            try:
                process_map = create_process_map(df, case_id, activity, resource)
                st.graphviz_chart(process_map)
                add_explanation("Process Map", "The process map visualizes the flow of activities. Nodes represent activities, while edges show transitions between activities. The thickness of edges indicates frequency of transitions. Start activities are in blue, end activities in pink.")
            except Exception as e:
                st.error(f"Error generating process map: {str(e)}")
                st.info("This might occur due to complexities in the process or limitations in the visualization library. Try filtering the data or adjusting the process map parameters.")

            # Variant Analysis
            st.subheader("Variant Analysis")
            try:
                variant_flow = create_variant_flow(df, case_id, activity)
                st.plotly_chart(variant_flow, use_container_width=True)

                # Display variant details in a table
                variants = df.groupby(case_id)[activity].agg(list).value_counts().reset_index()
                variants.columns = ['Variant', 'Frequency']
                variants['Variant'] = variants['Variant'].apply(lambda x: ' -> '.join(x))
                variants = variants.head(20)  # Show top 20 variants
                st.write("Top 20 Variants Details:")
                st.table(variants)
                
                add_explanation("Variant Analysis", "This visualization shows the most common process variants (unique sequences of activities). Each row represents a variant, with colored blocks indicating activities. The table below provides detailed frequency information for each variant.")
            except Exception as e:
                st.error(f"Error in variant analysis: {str(e)}")
                st.info("This might occur if there are issues with the case ID or activity columns. Please check your data.")

            # Activity Distribution
            st.subheader("Activity Distribution")
            try:
                activity_counts = df[activity].value_counts().reset_index()
                activity_counts.columns = ['Activity', 'Frequency']
                fig = px.bar(activity_counts, x='Activity', y='Frequency')
                fig.update_layout(xaxis_title="Activity", yaxis_title="Frequency")
                st.plotly_chart(fig, use_container_width=True)
                add_explanation("Activity Distribution", "This chart shows the frequency of each activity in the process. It helps identify the most common and least common activities, which can be useful for process optimization.")
            except Exception as e:
                st.error(f"Error generating activity distribution: {str(e)}")
                st.info("This might occur if there are issues with the activity column. Please check your data.")

            # Transition Matrix
            st.subheader("Transition Matrix")
            try:
                transition_matrix = create_transition_matrix(df, case_id, activity)
                fig = px.imshow(transition_matrix, labels=dict(x="Next Activity", y="Current Activity", color="Frequency"), 
                                x=transition_matrix.columns, y=transition_matrix.index)
                st.plotly_chart(fig, use_container_width=True)
                
                add_explanation("Transition Matrix", "The transition matrix shows the frequency of transitions between activities. Each cell represents the number of times one activity (row) is followed by another activity (column). Darker colors indicate more frequent transitions.")
            except ValueError as e:
                st.warning(f"Unable to create transition matrix. Error: {str(e)}")
                st.info("This might occur if there are cases with only one activity or if activities aren't properly paired.")

            # Resource Analysis (if resource column is selected)
            if resource:
                st.subheader("Resource Analysis")
                try:
                    resource_activity = df.groupby(resource)[activity].value_counts().unstack(fill_value=0)
                    fig = px.imshow(resource_activity, labels=dict(x="Activity", y="Resource", color="Frequency"))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    add_explanation("Resource Analysis", "This heatmap shows the frequency of activities performed by each resource. It helps identify specialization of resources and potential bottlenecks in the process.")
                except Exception as e:
                    st.error(f"Error in resource analysis: {str(e)}")
                    st.info("This might occur if there are issues with the resource or activity columns. Please check your data.")

    else:
        st.info("Please upload a CSV file to begin the analysis.")

if __name__ == "__main__":
    main()
