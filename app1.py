import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
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
                    dot.edge(str(source), str(target), label=f"{res}\n({count})", penwidth=str(0.5 + count/edge_counts.max()*2), fontsize='10', color='gray')
        else:
            edges = df.groupby(case_id)[activity].apply(lambda x: list(zip(x, x[1:]))).explode()
            edge_counts = edges.value_counts()
            for (source, target), count in edge_counts.items():
                dot.edge(str(source), str(target), label=str(count), penwidth=str(0.5 + count/edge_counts.max()*2), fontsize='10', color='gray')

        return dot
    except Exception as e:
        raise ValueError(f"Error in creating process map: {str(e)}")

# Function to create variant flow
def create_variant_flow(df, case_id, activity, top_n=20):
    variants = df.groupby(case_id)[activity].agg(list).value_counts()
    top_variants = variants.head(top_n)

    variant_data = []
    for i, (variant, count) in enumerate(top_variants.items(), 1):
        variant_name = f"Variant {i}<br>({' -> '.join(variant)})"
        for j, act in enumerate(variant):
            variant_data.append({
                'Variant': variant_name,
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
    transition_counts = transitions.explode().value_counts()
    transition_matrix = transition_counts.unstack(fill_value=0)
    return transition_matrix

# Function to add explanation
def add_explanation(title, explanation):
    return f"<span title='{explanation}'>ℹ️ {title}</span>"

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
            st.markdown(add_explanation("Process Overview", "This section provides key metrics about the process."), unsafe_allow_html=True)
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            col1.metric("Tickets", df[case_id].nunique())
            col2.metric("Activities", df[activity].nunique())
            col3.metric("Events", len(df))
            col4.metric("Users", df[resource].nunique() if resource else "N/A")
            col5.metric("Products", df['product'].nunique() if 'product' in df.columns else "N/A")
            col6.metric("Variants", df.groupby(case_id)[activity].agg(tuple).nunique())

            # Time range
            st.write(f"Time Range: {df[timestamp].min().date()} to {df[timestamp].max().date()}")

            # Number of cases per month
            st.markdown(add_explanation("Number of cases per month by start date", "This chart shows the distribution of cases over time."), unsafe_allow_html=True)
            df['month'] = df[timestamp].dt.to_period('M')
            cases_per_month = df.groupby('month')[case_id].nunique().reset_index()
            cases_per_month['month'] = cases_per_month['month'].astype(str)
            fig = px.bar(cases_per_month, x='month', y=case_id, labels={'month': 'Month', case_id: 'Number of Cases'})
            st.plotly_chart(fig, use_container_width=True)

            # Process Map
            st.markdown(add_explanation("Process Map", "The process map visualizes the flow of activities."), unsafe_allow_html=True)
            try:
                process_map = create_process_map(df, case_id, activity, resource)
                st.graphviz_chart(process_map)
            except Exception as e:
                st.error(f"Error generating process map: {str(e)}")
                st.info("This might occur due to complexities in the process or limitations in the visualization library. Try filtering the data or adjusting the process map parameters.")

            # Variant Analysis
            st.markdown(add_explanation("Variant Analysis", "This visualization shows the most common process variants (unique sequences of activities)."), unsafe_allow_html=True)
            try:
                variant_flow = create_variant_flow(df, case_id, activity)
                st.plotly_chart(variant_flow, use_container_width=True)
            except Exception as e:
                st.error(f"Error generating variant analysis: {str(e)}")

            # Transition Matrix
            st.markdown(add_explanation("Transition Matrix", "The transition matrix shows the frequency of transitions between activities."), unsafe_allow_html=True)
            try:
                transition_matrix = create_transition_matrix(df, case_id, activity)
                st.write(transition_matrix)
            except Exception as e:
                st.error(f"Error generating transition matrix: {str(e)}")

if __name__ == "__main__":
    main()
