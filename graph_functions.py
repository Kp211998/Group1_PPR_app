import time

import streamlit as st  # Importing the Streamlit library
import networkx as nx  # Importing NetworkX for graph manipulation
import graphviz  # Importing Graphviz for visualization
import uuid
import numpy as np
from networkx.algorithms import isomorphism
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import SpectralClustering
from pyvis.network import Network


# Function to output nodes and edges of a graph
def output_nodes_and_edges(graph: nx.Graph):
    st.write(list(graph.nodes))
    st.write(list(graph.edges))


# Function to count the number of nodes in a graph
def count_nodes(graph: nx.Graph):
    count = graph.number_of_nodes()
    st.info(f"Total number of nodes are {count}")


# Function to check if there is a path between two nodes in a graph
def check_path(graph: nx.Graph):
    node_list = st.session_state.get('graph_dict', {}).get('nodes', [])
    node_name_list = []
    for node_id in graph.nodes:
        for node in node_list:
            if node['id'] == node_id:
                node_name_list.append({'id': node['id'], 'label': node['data']['label']})

    def custom_format_func(option):
        return option["label"]

    node1_col, node2_col = st.columns(2)
    with node1_col:
        node1_select_label = st.selectbox("Select first node", node_name_list, format_func=custom_format_func,
                                          key="node1_select")
        node1_select = node1_select_label['id']
    with node2_col:
        node2_select_label = st.selectbox("Select second node",
                                          options=[node for node in node_name_list if node != node1_select_label],
                                          format_func=custom_format_func, key="node2_select")
        node2_select = node2_select_label['id']

    if node1_select and node2_select and nx.has_path(graph, node1_select, node2_select):
        st.success(
            f"There is a path between node {node1_select_label['label']} and node {node2_select_label['label']}.")
    else:
        st.error(f"There is no path between node {node1_select_label['label']} and node {node2_select_label['label']}.")


# Function to check if a graph is empty
def is_empty(graph: nx.Graph):
    is_graph_empty = nx.is_empty(graph)

    if is_graph_empty:
        st.info("The graph is empty.")
    else:
        st.info("The graph is not empty.")


# Function to check if a graph is directed
def is_directed(graph: nx.Graph):
    is_graph_directed = nx.is_directed(graph)
    if is_graph_directed:
        st.info("The graph is directed.")
    else:
        st.info("The graph is not directed")


# Function to display information about a specific node in a graph
def specific_node(graph: nx.Graph):
    node_list = st.session_state.get('graph_dict', {}).get('nodes', [])
    node_name_list = []
    for node_id in graph.nodes:
        for node in node_list:
            if node['id'] == node_id:
                node_name_list.append({'id': node['id'], 'label': node['data']['label']})

    def custom_format_func(option):
        return option["label"]

    node_select = st.selectbox(
        "Select the node label to see the details",
        node_name_list, format_func=custom_format_func
    )
    if node_select['id']:
        st.write(graph.nodes[node_select['id']])


# Function to display information about a specific edge in a graph
def specific_edge(graph: nx.Graph):
    node_list = st.session_state.get('graph_dict', {}).get('nodes', [])
    node_name_list = []
    for node_id in graph.nodes:
        for node in node_list:
            if node['id'] == node_id:
                node_name_list.append({'id': node['id'], 'label': node['data']['label']})

    def custom_format_func(option):
        return option["label"]

    node1_col, node2_col = st.columns(2)
    with node1_col:
        node1_select_label = st.selectbox("Select first node", node_name_list, format_func=custom_format_func,
                                          key="node1_select")
        node1_select = node1_select_label['id']
    with node2_col:
        node2_select_label = st.selectbox("Select second node",
                                          options=[node for node in node_name_list if node != node1_select_label],
                                          format_func=custom_format_func, key="node2_select")
        node2_select = node2_select_label['id']

    show_edge_button = st.button("Show edge details", use_container_width=True, type="primary")
    if show_edge_button:
        try:
            edge = graph.edges[node1_select, node2_select]
            if edge:
                st.write(edge)
        except KeyError:
            st.error(f"Edge {node1_select_label['label']} to {node2_select_label['label']} does not exist.")


# Function to calculate and display the density of a graph
def find_density(graph: nx.Graph):
    density = nx.density(graph)
    st.info(f"The density of graph is {density}")


# Function to calculate and display the shortest path between two nodes in a graph
def shortest_path(graph: nx.Graph):
    node_list = st.session_state.get('graph_dict', {}).get('nodes', [])
    edge_list = st.session_state.get('graph_dict', {}).get('edges', [])
    node_name_list = []
    for node_id in graph.nodes:
        for node in node_list:
            if node['id'] == node_id:
                node_name_list.append({'id': node['id'], 'label': node['data']['label']})

    def custom_format_func(option):
        return option["label"]

    node1_col, node2_col = st.columns(2)
    with node1_col:
        node1_select = st.selectbox("Select first node", node_name_list, format_func=custom_format_func,
                                    key="node1_select")
        node1_select_id = node1_select['id']
    with node2_col:
        node2_select = st.selectbox("Select second node",
                                    options=[node for node in node_name_list if node != node1_select],
                                    format_func=custom_format_func, key="node2_select")
        node2_select_id = node2_select['id']

    try:
        shortest_path_for_graph = nx.shortest_path(graph, node1_select_id, node2_select_id)
        shortest_path_text = []
        for node in node_list:
            for node_id in shortest_path_for_graph:
                if node_id == node['id']:
                    shortest_path_text.append(node['data']['label'])

        path_with_arrows = " --> ".join(shortest_path_text)
        st.success(
            f"The shortest path between {node1_select['label']} and {node2_select['label']} is '{path_with_arrows}'")
        subgraph = graph.subgraph(shortest_path_for_graph)
        st.write(subgraph)

        graphviz_graph = graphviz.Digraph()
        edge_label = ''
        for node_id in subgraph.nodes:
            graphviz_graph.node(node_id,
                                label=next((node['data']['label'] for node in node_list if node['id'] == node_id),
                                           None))
        for edge_ids in subgraph.edges:
            for edge in edge_list:
                if edge['source'] == edge_ids[0] and edge['target'] == edge_ids[1]:
                    edge_label = edge['label']
            graphviz_graph.edge(edge_ids[0], edge_ids[1], edge_label)
        st.graphviz_chart(graphviz_graph)

    except nx.NetworkXNoPath:
        st.error(f"There is no path between {node1_select} and {node2_select}")


# Function to calculate and display the shortest paths from a selected start node to all other nodes in a graph
def show_shortest_paths(graph: nx.DiGraph):  # Rectified code of Prof. Luder
    try:
        graph_dict_tree = st.session_state["graph_dict"]
        node_list_tree = graph_dict_tree["nodes"]
        edge_list_tree = graph_dict_tree["edges"]
        node_list_tree_found = []
        edge_list_tree_found = []
        node_name_list_tree = [node["id"] for node in node_list_tree]

        start_node_select_tree = st.selectbox(
            "Select the start node of the shortest paths",
            options=node_name_list_tree
        )

        # Remove the selected start node from the options for target node selection
        target_node_options = [node for node in node_name_list_tree if node != start_node_select_tree]
        target_node_select_tree = st.selectbox(
            "Select the target node of the shortest paths",
            options=target_node_options
        )

        is_tree_button = st.button("Calculate trees", use_container_width=True, type="primary")

        if is_tree_button:
            tree_list = nx.shortest_path(graph, source=start_node_select_tree, target=target_node_select_tree,
                                         weight="dist")

            if not tree_list:
                st.write(f"There is no tree starting from {start_node_select_tree}.")
            else:
                for tree in tree_list:
                    st.write(f"The node {tree} is a member of the tree")
                    for node_element in node_list_tree:
                        if node_element["id"] == tree:
                            to_be_assigned_element = node_element
                            if to_be_assigned_element not in node_list_tree_found:
                                node_list_tree_found.append(node_element)

                for edge_element in edge_list_tree:
                    for source_node in node_list_tree_found:
                        for sink_node in node_list_tree_found:
                            if edge_element["source"] == source_node["id"] and edge_element["target"] == \
                                    sink_node["id"]:
                                edge_list_tree_found.append(edge_element)

                show_graph_without_weights(node_list_tree_found, edge_list_tree_found)
    except nx.NetworkXNoPath:
        st.error(f"There is no path between {start_node_select_tree} and {target_node_select_tree}")


# Function to display the graph without considering the weights of the edges
def show_graph_without_weights(nodes, edges):
    def set_color(node_type):
        color = "Grey"
        if node_type == "Person":
            color = "Blue"
        elif node_type == "Node":
            color = "Green"
        return color

    graph = graphviz.Digraph()
    for node in nodes:
        node_name = node["id"]
        graph.node(node_name, color=set_color(node["type"]))
    for edge in edges:
        source = edge["source"]
        target = edge["target"]
        label = edge.get("label", "")  # Changed from "type" to "label"
        graph.edge(source, target, label)
    st.graphviz_chart(graph)


def minimum_spanning_tree(graph: nx.Graph):
    G = nx.Graph()
    edge_list = st.session_state["edge_list"]
    node_list = st.session_state["node_list"]
    for edge in edge_list:
        G.add_edge(edge["source"], edge["target"])
    minimum_spanning_tree_graph = nx.minimum_spanning_tree(G)
    graphviz_graph = graphviz.Digraph()
    edge_label = ''
    for node_id in minimum_spanning_tree_graph.nodes:
        graphviz_graph.node(node_id,
                            label=next((node['data']['label'] for node in node_list if node['id'] == node_id), None))
    for edge_ids in minimum_spanning_tree_graph.edges:
        for edge in edge_list:
            if edge['source'] == edge_ids[0] and edge['target'] == edge_ids[1]:
                edge_label = edge['label']
        graphviz_graph.edge(edge_ids[0], edge_ids[1], edge_label)
    st.graphviz_chart(graphviz_graph)


def spanning_tree(graph: nx.Graph):
    node_list = st.session_state.get('graph_dict', {}).get('nodes', [])
    node_name_list = []
    for node_id in graph.nodes:
        for node in node_list:
            if node['id'] == node_id:
                node_name_list.append({'id': node['id'], 'label': node['data']['label']})

    def custom_format_func(option):
        return option['data']["label"]

    root_node = st.selectbox(
        "Select the start node of the shortest paths",
        node_list, format_func=custom_format_func
    )
    root_node_id = root_node['id']

    G = nx.Graph()
    edge_list = st.session_state["edge_list"]
    for edge in edge_list:
        G.add_edge(edge["source"], edge["target"])
    spanning_tree_graph = nx.dfs_tree(G, source=root_node_id)

    graphviz_graph = graphviz.Digraph()
    edge_label = ''
    for node_id in spanning_tree_graph.nodes:
        graphviz_graph.node(node_id,
                            label=next((node['data']['label'] for node in node_list if node['id'] == node_id), None))
    for edge_ids in spanning_tree_graph.edges:
        for edge in edge_list:
            if edge['source'] == edge_ids[0] and edge['target'] == edge_ids[1]:
                edge_label = edge['label']
        graphviz_graph.edge(edge_ids[0], edge_ids[1], edge_label)
    st.graphviz_chart(graphviz_graph)


def system_analysis(graph: nx.DiGraph):
    if 'view_graphs' not in st.session_state:
        st.session_state['view_graphs'] = []

    node_list = st.session_state["node_list"]
    edge_list = st.session_state["edge_list"]
    analysis_list = ["Impact of input product on process step", "Impact of process on process",
                     "Dependencies between resources step", "Recurring components"]

    product_list = []
    process_list = []
    resources_list = []
    for node in node_list:
        if node['type'] == 'product':
            product_list.append(node)
        elif node['type'] == 'process':
            process_list.append(node)
        elif node['type'] == 'resource':
            resources_list.append(node)

    def method_changed():
        st.session_state.show_save_impact = False
        st.session_state.find_recurring_components_clicked = False

    selected_method = st.selectbox("Select the analysis method", options=analysis_list, on_change=method_changed)
    selectbox_list_1 = []
    selectbox_list_2 = []
    if selected_method == "Impact of input product on process step":
        selectbox_list_1 = product_list
        selectbox_list_2 = process_list
    elif selected_method == "Impact of process on process":
        selectbox_list_1 = process_list
        selectbox_list_2 = process_list
    elif selected_method == "Dependencies between resources step":
        selectbox_list_1 = resources_list
        selectbox_list_2 = resources_list

    if selected_method != "Recurring components":
        node1_select = None
        node2_select = None

        def custom_format_func(option):
            return option['data']["label"]

        node1_col, node2_col = st.columns(2)
        with node1_col:
            node1_select_label = st.selectbox("Select first node", selectbox_list_1, format_func=custom_format_func,
                                              key="node1_select")
            if node1_select_label:
                node1_select = node1_select_label['id']
        with node2_col:
            node2_select_label = st.selectbox("Select second node",
                                              options=[node for node in selectbox_list_2 if node != node1_select_label],
                                              format_func=custom_format_func, key="node2_select")
            if node2_select_label:
                node2_select = node2_select_label['id']

        if 'show_save_impact' not in st.session_state:
            st.session_state.show_save_impact = False

        def check_impact_clicked():
            st.session_state.show_save_impact = True

        check_impact = st.button("check impact", key="check_impact", use_container_width=True,
                                 disabled=node1_select is None or node2_select is None,
                                 on_click=check_impact_clicked) is None

        if st.session_state['show_save_impact']:
            if node1_select and node2_select and nx.has_path(graph, node1_select, node2_select):
                st.success(f"{node1_select_label['data']['label']} has impact on {node2_select_label['data']['label']}")
                shortest_path_for_graph = nx.shortest_path(graph, node1_select, node2_select)
                subgraph = graph.subgraph(shortest_path_for_graph)
                graphviz_graph = graphviz.Digraph()
                edge_label = ''
                for node_id in subgraph.nodes:
                    graphviz_graph.node(node_id,
                                        label=next(
                                            (node['data']['label'] for node in node_list if node['id'] == node_id),
                                            None))
                for edge_ids in subgraph.edges:
                    for edge in edge_list:
                        if edge['source'] == edge_ids[0] and edge['target'] == edge_ids[1]:
                            edge_label = edge['label']
                    graphviz_graph.edge(edge_ids[0], edge_ids[1], edge_label)
                st.graphviz_chart(graphviz_graph)
                relation_list = []
                st.subheader("Save this impact as a relation")

                def selectbox_change():
                    st.session_state.show_save_impact = True

                selected_view = st.selectbox("Select View for Impact",
                                             options=["Mechanical View", "Basic Engineering View",
                                                      "Sustainability View"], on_change=selectbox_change)
                if selected_view == "Mechanical View":
                    relation_list = ["Part of", "Assembled with"]
                elif selected_view == "Basic Engineering View":
                    relation_list = ["Part of", "Impact on"]
                elif selected_view == "Sustainability View":
                    relation_list = ["Impacts", "Powers"]
                selected_impact_relation = st.selectbox("Select relationship", relation_list,
                                                        on_change=selectbox_change)
                if selected_impact_relation:

                    save_impact = st.button(label="Save Impact", key="save_impact", use_container_width=True)

                    if node1_select and node2_select and save_impact:
                        save_edge(node1_select_label['id'], selected_impact_relation, node2_select_label['id'],
                                  selected_view)
                        time.sleep(1)
                        st.rerun()
            else:
                st.error(
                    f"{node1_select_label['data']['label']} does not have any impact on {node2_select_label['data']['label']}")

    else:
        node1_select = None
        node2_select = None

        def custom_format_func(option):
            return option['data']["label"]

        def node_changed():
            st.session_state.find_recurring_components_clicked = False

        node1_col, node2_col = st.columns(2)
        with node1_col:
            node1_select_label = st.selectbox("Select first node", node_list,
                                              format_func=custom_format_func,
                                              key="node1_select", on_change=node_changed)
            if node1_select_label:
                node1_select = node1_select_label['id']
        with node2_col:
            node2_select_label = st.selectbox("Select second node",
                                              options=[node for node in node_list if
                                                       node != node1_select_label],
                                              format_func=custom_format_func, key="node2_select",
                                              on_change=node_changed)
            if node2_select_label:
                node2_select = node2_select_label['id']

        if 'find_recurring_components_clicked' not in st.session_state:
            st.session_state.find_recurring_components_clicked = False

        def click_find_recurring_components_button():
            st.session_state.find_recurring_components_clicked = True

        find_recurring_components = st.button("Find Recurring Components", key="find_recurring_components",
                                              use_container_width=True, on_click=click_find_recurring_components_button,
                                              disabled=node1_select is None or node2_select is None) is None

        if node1_select and node2_select and st.session_state.find_recurring_components_clicked:
            successors_for_node_1 = list(graph.successors(node1_select))
            nodes_to_include_in_sub_1 = [node1_select] + successors_for_node_1
            subgraph_for_node_1 = graph.subgraph(nodes_to_include_in_sub_1)
            successors_for_node_2 = list(graph.successors(node2_select))
            nodes_to_include_in_sub_2 = [node2_select] + successors_for_node_2
            subgraph_for_node_2 = graph.subgraph(nodes_to_include_in_sub_2)
            g1 = nx.DiGraph(subgraph_for_node_1)
            g2 = nx.DiGraph(subgraph_for_node_2)

            graph_matcher = isomorphism.GraphMatcher(g1, g2)
            st.write(graph_matcher.is_isomorphic())
            similar_structures = [subgraph for subgraph in graph_matcher.subgraph_isomorphisms_iter()]

            if similar_structures:
                st.info(f"There can be {len(similar_structures)} mappings possible ")
                with st.expander("Isomorphic Mappings",expanded=False):
                    for idx, mapping in enumerate(similar_structures, start=1):
                        st.subheader(f"Mapping {idx}:")
                        for node_1, node_2 in mapping.items():
                            st.write(f"Node {node_1} ---> Node {node_2} ")
                        st.markdown("---")
            else:
                st.error("No similar components")


def save_edge(source_node, relation, target_node, selected_view):
    node_list = st.session_state["node_list"]

    edge_dict = {
        "id": uuid.uuid4().hex,
        "source": source_node,
        "target": target_node,
        "label": relation,
    }

    # Check if the edge already exists
    existing_edges = st.session_state.get("existing_edges", [])
    if (source_node, relation, target_node) in existing_edges:
        st.error("Edge already exists.")
        return

    source_node_data = {}
    target_node_data = {}
    for node in node_list:
        if node["id"] == source_node:
            source_node_data = node
        elif node["id"] == target_node:
            target_node_data = node

    # Get the selected view name
    selected_view_graph = selected_view + " Graph"

    # If the selected view name is not None and the graph exists in graph_dict
    if selected_view_graph + "_Nodes" not in st.session_state:
        st.session_state[f"{selected_view_graph}_Nodes"] = []
    if selected_view_graph + "_Edges" not in st.session_state:
        st.session_state[f"{selected_view_graph}_Edges"] = []
    if selected_view_graph not in st.session_state:
        st.session_state[selected_view_graph] = []
    if 'view_graphs' not in st.session_state:
        st.session_state['view_graphs'] = []

    if source_node not in st.session_state[f"{selected_view_graph}_Nodes"]:
        st.session_state[f"{selected_view_graph}_Nodes"].append(source_node_data)

    if target_node not in st.session_state[f"{selected_view_graph}_Nodes"]:
        st.session_state[f"{selected_view_graph}_Nodes"].append(target_node_data)

    st.session_state[f"{selected_view_graph}_Edges"].append(edge_dict)

    # Update existing edges
    existing_edges.append((source_node, relation, target_node))
    st.session_state["existing_edges"] = existing_edges

    graph_dict = {
        "nodes": st.session_state[f"{selected_view_graph}_Nodes"],
        "edges": st.session_state[f"{selected_view_graph}_Edges"],
    }
    st.session_state[selected_view_graph] = graph_dict
    view_graph_dict = {selected_view_graph: st.session_state[selected_view_graph]}
    st.session_state['view_graphs'].append(view_graph_dict)
    st.success("Impact edge saved successfully")
    st.session_state.show_save_impact = False


def spectral_clustering(graph, num_clusters):
    adjacency_matrix = nx.adjacency_matrix(graph).todense()
    spectral = SpectralClustering(n_clusters=num_clusters, affinity='precomputed', random_state=42)
    labels = spectral.fit_predict(adjacency_matrix)
    net = Network(height="500px", width="100%", notebook=False)
    net.barnes_hut()

    # Add nodes
    for node, label in enumerate(labels):
        net.add_node(node, label=str(node), color=label)

    # Add edges
    for edge in graph.edges():
        net.add_edge(edge[0], edge[1])

    # Show the network
    net.show("clustered_graph.html")
    st.components.v1.html(open("clustered_graph.html", "r").read(), width=800, height=600)

    # return labels

# def visualize_clusters(graph, labels):
