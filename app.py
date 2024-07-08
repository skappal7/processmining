import streamlit as st
import pandas as pd
import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.statistics.traces.generic.log import case_statistics
from pm4py.algo.organizational_mining.roles import algorithm as roles_discovery
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
import networkx as nx
import matplotlib.pyplot as plt
import io
from collections import defaultdict
from datetime import datetime

# Set page config
st.set_page_config(page_title="Advanced Process Mining App", layout="wide")

# Custom CSS for styling
st.markdown("""
<style>
.stButton>button {
    background-color: #4CAF50;
    color: white;
}
.stSelectbox {
    color: #4CAF50;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_log(file):
    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
        return df
    elif file.name.endswith('.xes'):
        log = xes_importer.apply(file)
        return log
    else:
        raise ValueError("Unsupported file format. Please upload a CSV or XES file.")

def convert_df_to_event_log(df, case_id, activity, timestamp, resource=None):
    df = df.rename(columns={case_id: 'case:concept:name', 
                            activity: 'concept:name', 
                            timestamp: 'time:timestamp'})
    if resource:
        df = df.rename(columns={resource: 'org:resource'})
    df['time:timestamp'] = pd.to_datetime(df['time:timestamp'])
    event_log = log_converter.apply(df)
    return event_log

def visualize_process_model(log, algorithm):
    if algorithm == "Alpha Miner":
        net, initial_marking, final_marking = pm4py.discover_petri_net_alpha(log)
    elif algorithm == "Heuristics Miner":
        net, initial_marking, final_marking = pm4py.discover_petri_net_heuristics(log)
    else:
        tree = inductive_miner.apply(log)
        net, initial_marking, final_marking = pm4py.convert_to_petri_net(tree)

    return visualize_using_networkx(net)

def visualize_using_networkx(net):
    G = nx.DiGraph()
    for place in net.places:
        G.add_node(place.name, node_type='place')
    for transition in net.transitions:
        G.add_node(transition.name, node_type='transition')
    for arc in net.arcs:
        G.add_edge(arc.source.name, arc.target.name)

    pos = nx.spring_layout(G)
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=500, font_size=8, arrows=True)
    nx.draw_networkx_labels(G, pos)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf

def analyze_process(log):
    variants_count = case_statistics.get_variant_statistics(log)
    variants_df = pd.DataFrame(variants_count).sort_values("count", ascending=False)

    activities = pm4py.get_event_attribute_values(log, "concept:name")
    activities_df = pd.DataFrame.from_dict(activities, orient='index', columns=['frequency']).sort_values('frequency', ascending=False)

    try:
        roles = roles_discovery.apply(log)
        roles_df = pd.DataFrame([(k, ', '.join(v)) for k, v in roles.items()], columns=['Resource', 'Activities'])
    except Exception as e:
        st.warning(f"Unable to perform role discovery. Error: {str(e)}")
        roles_df = None

    return variants_df, activities_df, roles_df

def perform_conformance_checking(log, net, initial_marking, final_marking):
    replayed_traces = token_replay.apply(log, net, initial_marking, final_marking)
    fitness = sum(trace['trace_fitness'] for trace in replayed_traces) / len(replayed_traces)
    return fitness

def create_dotted_chart(log):
    case_events = defaultdict(list)
    for trace in log:
        case_id = trace.attributes['concept:name']
        for event in trace:
            timestamp = event['time:timestamp'].timestamp()
            case_events[case_id].append(timestamp)
    
    plt.figure(figsize=(12, 6))
    for case_id, events in case_events.items():
        plt.plot(events, [case_id] * len(events), 'o')
    
    plt.xlabel('Timestamp')
    plt.ylabel('Case ID')
    plt.title('Dotted Chart')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf

def create_social_network(log):
    handover = defaultdict(lambda: defaultdict(int))
    for trace in log:
        for i in range(len(trace) - 1):
            resource1 = trace[i].get('org:resource', 'Unknown')
            resource2 = trace[i+1].get('org:resource', 'Unknown')
            handover[resource1][resource2] += 1

    G = nx.DiGraph()
    for resource1, targets in handover.items():
        for resource2, weight in targets.items():
            G.add_edge(resource1, resource2, weight=weight)

    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=8, arrows=True)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf

def main():
    st.title("Advanced Process Mining App")

    st.sidebar.title("Controls")
    uploaded_file = st.sidebar.file_uploader("Upload event log (CSV or XES)", type=["csv", "xes"])

    if uploaded_file is not None:
        data = load_log(uploaded_file)

        if isinstance(data, pd.DataFrame):
            st.sidebar.subheader("Column Selection")
            case_id = st.sidebar.selectbox("Select case ID column", data.columns)
            activity = st.sidebar.selectbox("Select activity column", data.columns)
            timestamp = st.sidebar.selectbox("Select timestamp column", data.columns)
            resource = st.sidebar.selectbox("Select resource column (optional)", ['None'] + list(data.columns))

            if st.sidebar.button("Process Data"):
                with st.spinner("Processing data..."):
                    try:
                        resource = resource if resource != 'None' else None
                        log = convert_df_to_event_log(data, case_id, activity, timestamp, resource)
                        st.session_state['log'] = log
                        st.success("Data processed successfully!")
                    except Exception as e:
                        st.error(f"Error processing data: {str(e)}")
                        st.session_state['log'] = None
        else:
            log = data
            st.session_state['log'] = log
            st.success("XES file loaded successfully!")

        if 'log' in st.session_state and st.session_state['log'] is not None:
            log = st.session_state['log']

            st.sidebar.subheader("Process Discovery")
            algorithm = st.sidebar.selectbox("Select mining algorithm", ["Alpha Miner", "Heuristics Miner", "Inductive Miner"])

            if st.sidebar.button("Discover Process Model"):
                with st.spinner("Discovering process model..."):
                    img = visualize_process_model(log, algorithm)
                    st.subheader("Process Model")
                    st.image(img)

            if st.sidebar.button("Analyze Process"):
                with st.spinner("Analyzing process..."):
                    variants_df, activities_df, roles_df = analyze_process(log)

                    st.subheader("Variant Analysis")
                    st.dataframe(variants_df)

                    st.subheader("Activity Frequency")
                    st.bar_chart(activities_df['frequency'])

                    if roles_df is not None:
                        st.subheader("Resource Roles")
                        st.dataframe(roles_df)

            if st.sidebar.button("Perform Conformance Checking"):
                with st.spinner("Performing conformance checking..."):
                    net, initial_marking, final_marking = pm4py.discover_petri_net_inductive(log)
                    fitness = perform_conformance_checking(log, net, initial_marking, final_marking)
                    st.subheader("Conformance Checking")
                    st.write(f"Fitness: {fitness:.2f}")

            if st.sidebar.button("Generate Advanced Visualizations"):
                with st.spinner("Generating advanced visualizations..."):
                    st.subheader("Dotted Chart")
                    dotted_chart = create_dotted_chart(log)
                    st.image(dotted_chart)

                    st.subheader("Social Network")
                    social_network = create_social_network(log)
                    st.image(social_network)

            st.sidebar.subheader("Filtering")
            activities = pm4py.get_event_attribute_values(log, "concept:name")
            selected_activities = st.sidebar.multiselect("Select activities to include", list(activities.keys()))

            if selected_activities:
                filtered_log = pm4py.filter_event_attribute_values(log, "concept:name", selected_activities, level="event")
                st.sidebar.text(f"Filtered log: {len(filtered_log)} traces")

                if st.sidebar.button("Apply Filter"):
                    with st.spinner("Applying filter and updating visualizations..."):
                        img = visualize_process_model(filtered_log, algorithm)
                        st.subheader("Filtered Process Model")
                        st.image(img)

                        variants_df, activities_df, roles_df = analyze_process(filtered_log)
                        
                        st.subheader("Filtered Variant Analysis")
                        st.dataframe(variants_df)

                        st.subheader("Filtered Activity Frequency")
                        st.bar_chart(activities_df['frequency'])

                        if roles_df is not None:
                            st.subheader("Filtered Resource Roles")
                            st.dataframe(roles_df)

        else:
            st.warning("Please process the data first by selecting the appropriate columns and clicking 'Process Data'.")

    else:
        st.info("Please upload a CSV or XES file to begin process mining.")

if __name__ == "__main__":
    main()
