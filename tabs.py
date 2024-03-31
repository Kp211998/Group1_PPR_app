import streamlit as st  # Importing the Streamlit library
import uuid  # Importing UUID for generating unique identifiers
import graphviz  # Importing Graphviz for visualization
from streamlit_option_menu import option_menu  # Importing a custom module for creating an option menu
from streamlit_agraph import agraph, Node, Edge, Config  # Importing agraph for graph visualization
import networkx as nx  # Importing NetworkX for graph analysis
from graph_functions import (output_nodes_and_edges, count_nodes, is_empty, find_density, is_directed, check_path,
                             specific_node, specific_edge, shortest_path,
                             spanning_tree,
                             minimum_spanning_tree, system_analysis,
                             spectral_clustering)  # Importing custom graph functions
import time
import json
from st_cytoscape import cytoscape
import pandas as pd

type_list = ["product", "process", "resource"]


# Function to upload graph from JSON
def upload_graph():
    uploaded_nodes = []
    uploaded_edges = []
    uploaded_graph = st.file_uploader("Upload an existing graph", type="json")  # Uploading JSON file

    if uploaded_graph is not None:
        uploaded_graph_dict = json.load(uploaded_graph)

        if "nodes" in uploaded_graph_dict and "edges" in uploaded_graph_dict:
            uploaded_nodes = uploaded_graph_dict["nodes"]
            uploaded_edges = uploaded_graph_dict["edges"]

        else:
            st.error("Invalid JSON format. Please ensure that the JSON file contains 'nodes' and 'edges' keys.")
            return

    else:
        st.info("Please upload a graph if available.")

    update_graph_button = st.button("Update Graph", use_container_width=True, type="primary",
                                    disabled=uploaded_graph is None)  # Button to update graph

    if update_graph_button:
        st.session_state["node_list"] = uploaded_nodes
        st.session_state["edge_list"] = uploaded_edges
        update_graph_dict()
        st.success("Graph uploaded successfully.")
        st.rerun()
    st.text("You can click on the below button to create a PPR model and download the JSON file")
    if 'app_visible' not in st.session_state:
        st.session_state.app_visible = False
    # Button to toggle the embedded app visibility
    if not st.session_state.app_visible:
        if st.button("Open MAPPR", use_container_width=True):
            st.session_state.app_visible = True
            st.rerun()
    else:
        if st.button("Close MAPPR", use_container_width=True):
            st.session_state.app_visible = False
            st.rerun()

    # Display the embedded app if it's visible
    if st.session_state.app_visible:
        iframe_code = '<iframe src="https://dhoffmannovgu.github.io/PPR-Editor/" width="100%" height="600"></iframe>'
        st.markdown(iframe_code, unsafe_allow_html=True)


def update_graph_dict():
    graph_dict = {
        "nodes": st.session_state["node_list"],
        "edges": st.session_state["edge_list"],
    }
    st.session_state["graph_dict"] = graph_dict


def node_functions():
    tab_list = [
        "Create Node",
        "Update Node",
        "Delete Node"
    ]
    selected = option_menu("Node Operations", tab_list,
                           menu_icon="cast",
                           default_index=0,
                           orientation="horizontal"
                           )

    if selected == "Create Node":
        create_node()
    elif selected == "Update Node":
        update_node()  # Function call to update node
    elif selected == "Delete Node":
        delete_node()
def create_node():
    st.header("Create Node")
    # Node type selection box
    type_node = st.selectbox("Select node type", type_list)

    # Input for node label
    label = st.text_input("Enter label for the node")

    # Adding views
    if 'view_names' not in st.session_state:
        st.session_state.view_names = []
    if 'views' not in st.session_state:
        st.session_state.views = []

    st.header("Add Views")
    selected_view = st.text_input("Enter name for the view")
    add_view_button = st.button("Add View")
    if add_view_button and selected_view not in st.session_state.view_names:
        st.session_state.view_names.append(selected_view)
        selected_view = ""  # Clear text input after adding view

    # Add Attributes section if views are added
    if st.session_state.view_names:
        st.header("Add Attributes")
        selected_view = st.selectbox("Select View", options=st.session_state.view_names)

        # Add Attribute form
        with st.form("add_attr_form", clear_on_submit=True):
            property_name = st.text_input(f"Enter property name")
            target_value = st.text_input(f"Enter target value ")
            min_value = st.text_input(f"Enter minimum value ")
            max_value = st.text_input(f"Enter maximum value")
            unit = st.text_input(f"Enter unit for property")

            # Check if the form is submitted
            if st.form_submit_button("Submit"):
                # Call the add_attribute_data function with the entered values
                add_attribute_data(property_name, target_value, min_value, max_value, unit, selected_view)

    # Save node button (disabled if label is not provided)
    def on_save_node_click():
        save_node(type_node, label, st.session_state['views'])

    save_node_btn_disabled = not label  # Disable button if label is not provided
    save_node_btn = st.button("Save Node", key=str(uuid.uuid4()), help="Click to save node",
                              on_click=on_save_node_click, disabled=save_node_btn_disabled)


def add_attribute_data(property_name, target_value, min_value, max_value, unit, view_name):
    data = {
        "name": property_name,
        "target_value": target_value,
        "min_value": min_value,
        "max_value": max_value,
        "unit": unit
    }
    for view in st.session_state['views']:
        if view['view_name'] == view_name:
            if 'properties' not in view:
                view['properties'] = []
            view['properties'].append(data)
            break
    else:
        view = {
            "view_name": view_name,
            "properties": [data]
        }
        st.session_state['views'].append(view)


def save_node(type_node, label, views):
    st.session_state["tab_index"] = 1
    formatted_views = {}
    for view in views:
        view_name = view['view_name']
        properties = []
        for prop in view['properties']:
            # Generate a unique ID for each property
            prop_id = str(uuid.uuid4())
            prop_data = {
                "name": prop['name'],
                "id": prop_id,
                "target_value": prop['target_value'],
                "min_value": prop['min_value'],
                "max_value": prop['max_value'],
                "unit": prop['unit']
            }
            properties.append(prop_data)
        formatted_views[view_name] = {
            "id": view.get('id', str(uuid.uuid4())),  # Generate a unique ID for each view if not provided
            "properties": properties
        }

    node = {
        "id": str(uuid.uuid4()),  # Generate a unique ID for the node
        "type": type_node,
        "data": {
            "label": label,
            "props": {
                "views": formatted_views
            }
        }
    }
    st.session_state["node_list"].append(node)
    st.session_state["views"] = []
    st.session_state["view_names"] = []


#
#
# Function to update a node
def update_node():
    node_list = st.session_state["node_list"]
    node_labels = [node["data"]["label"] for node in node_list]

    try:
        # Select the node to update
        node_to_update = st.selectbox("Select node to update", options=node_labels)

        # Find the index of the selected node in the list
        selected_index = node_labels.index(node_to_update)
        selected_node = node_list[selected_index]

        # Display current node properties
        st.write(f"Current properties of node '{node_to_update}':")

        # Allow users to update node properties
        new_label = st.text_input("Enter new label for the node", value=selected_node["data"]["label"])
        new_type = st.selectbox("Select new type for the node", options=type_list)

        # Add views
        st.header("Update Views")
        updated_views = {}
        selected_view_to_edit = st.selectbox("Select view to edit", options=selected_node["data"]["props"]["views"])
        propety_list = []
        view_id = None
        for view_name, view_data in selected_node["data"]["props"]["views"].items():
            if view_name == selected_view_to_edit:
                view_id = view_data['id']
                for prop in view_data["properties"]:
                    propety_list.append(prop['name'])
        selected_prop_to_edit = st.selectbox("Select prop to edit", options=propety_list)
        selected_index_for_prop = None
        for view_name, view_data in selected_node["data"]["props"]["views"].items():
            updated_properties = []
            for prop in view_data["properties"]:
                if prop["name"] == selected_prop_to_edit:
                    selected_index_for_prop = view_data["properties"].index(prop)
                    property_name = prop["name"]
                    target_value = st.text_input(f"Enter target value for property {property_name}",
                                                 value=prop["target_value"])
                    min_value = st.text_input(f"Enter minimum value for property {property_name}",
                                              value=prop["min_value"])
                    max_value = st.text_input(f"Enter maximum value for property {property_name}",
                                              value=prop["max_value"])
                    unit = st.text_input(f"Enter unit for property {property_name}", value=prop["unit"])
                    updated_properties.append({
                        "name": property_name,
                        "target_value": target_value,
                        "min_value": min_value,
                        "max_value": max_value,
                        "unit": unit
                    })

            # updated_views=node_list[selected_index]["data"]["props"]["views"]
            updated_views[selected_view_to_edit] = {
                "id": view_id,
                "properties": node_list[selected_index]["data"]["props"]["views"][selected_view_to_edit]["properties"]
            }
            if updated_properties is not None and len(updated_properties) > 0:
                updated_views[selected_view_to_edit]["properties"][selected_index_for_prop] = updated_properties[0]

        update_node_button = st.button("Update Node", key="update_node_button", help="Click to update node")

        if update_node_button:
            # Update node properties
            node_list[selected_index]["data"]["label"] = new_label
            node_list[selected_index]["type"] = new_type
            node_list[selected_index]["data"]["props"]["views"] = updated_views

            # Update session state with the modified node list
            st.session_state["node_list"] = node_list

            st.success(f"Node '{node_to_update}' has been updated.")

    except ValueError:
        st.error("There are no nodes added yet. Please create nodes or import a graph")


def delete_node():
    node_list = st.session_state["node_list"]
    node_names = [node["data"]['label'] for node in node_list]

    node_to_delete = st.selectbox("Select node to delete", options=node_names)
    delete_node_button = st.button("Delete Node", key="delete_node_button", use_container_width=True, type="primary")

    if delete_node_button:
        # Remove the node from the node list
        st.session_state["node_list"] = [node for node in node_list if node["data"]['label'] != node_to_delete]

        # Remove edges connected to the deleted node from the edge list
        # st.session_state["edge_list"] = [edge for edge in st.session_state["edge_list"]
        #                                  if edge["source"] != node_to_delete and edge["target"] != node_to_delete]

        st.session_state["deleted_node"] = node_to_delete  # Store the deleted node name

        st.success(f"Node '{node_to_delete}' has been deleted.")
        time.sleep(1)
        st.experimental_rerun()


# Function to create a relation
def edge_functions():
    tab_list = [
        "Create Relations",
        "Update Relations",
        "Delete Relations"
    ]
    selected = option_menu("Edge Operations", tab_list,
                           menu_icon="cast",
                           default_index=0,
                           orientation="horizontal"
                           )

    if selected == "Create Relations":
        create_relation()  # Function call to create relations
    elif selected == "Update Relations":
        update_edge()  # Function call to update relations
    elif selected == "Delete Relations":
        delete_edge()  # Function call to delete relations


def create_relation():
    node_list = st.session_state["node_list"]
    node_name_list = [node["data"]['label'] for node in node_list]
    source_node = None
    target_node = None

    def custom_format_func(option):
        return option['data']["label"]

    node1_col, relation_col, node2_col = st.columns(3)
    with node1_col:
        node1_select_label = st.selectbox("Select first node", node_list, format_func=custom_format_func,
                                          key="node1_select")
        if node1_select_label:
            source_node = node1_select_label['id']
    with relation_col:
        relation_list = ["Connected to", "output of", 'Input for', "Part of", "Assembled with"]
        relation_name = st.selectbox("Specify the relation", options=relation_list)
    with node2_col:
        node2_select_label = st.selectbox("Select second node",
                                          options=[node for node in node_list if node != node1_select_label],
                                          format_func=custom_format_func, key="node2_select")
        if node2_select_label:
            target_node = node2_select_label['id']

    if source_node == target_node:
        st.warning("Please select two different nodes.")
    else:
        existing_edges = st.session_state["edge_list"]
        edge_exists = any(
            edge["source"] == source_node and edge["target"] == target_node and edge["label"] == relation_name
            for edge in existing_edges
        )
        if edge_exists:
            st.error(
                f"A relation of type '{relation_name}' already exists between '{node1_select_label['data']['label']}' and '{node2_select_label['data']['label']}'.")
        else:
            store_edge_button = st.button("Save Relationship", use_container_width=True, type="primary")
            if store_edge_button:
                save_edge(source_node, relation_name, target_node)
                st.write(
                    f"{node1_select_label['data']['label']} is {relation_name} {node2_select_label['data']['label']}")


# Function to save edge details
def save_edge(source_node, relation, target_node):
    edge_dict = {
        "id": uuid.uuid4().hex,
        "source": source_node,
        "target": target_node,
        "label": relation,
    }
    st.session_state["edge_list"].append(edge_dict)


# Function to update an edge
def update_edge():
    edge_list = st.session_state["edge_list"]
    node_list = st.session_state["node_list"]
    edge_descriptions = []

    try:
        source_node_label = ''
        target_node_label = ''
        for edge in edge_list:
            for node in node_list:
                if node["id"] == edge["source"]:
                    source_node_label = node['data']['label']
                elif node["id"] == edge["target"]:
                    target_node_label = node['data']['label']
            edge_descriptions.append(f"{source_node_label} -> {target_node_label} ({edge['label']})")

        # Select the edge to update
        selected_edge_description = st.selectbox("Select edge to update", options=edge_descriptions)

        # Find the index of the selected edge in the list
        selected_index = edge_descriptions.index(selected_edge_description)
        selected_edge = edge_list[selected_index]

        # Display current edge properties
        st.write(f"Current properties of edge '{selected_edge_description}':")

        def custom_format_func(option):
            return option['data']["label"]

        node1_select = st.selectbox("Select new source node for the edge", node_list, format_func=custom_format_func,
                                    key="node1_select")
        new_source = node1_select['id']
        node2_select = st.selectbox("Select new target node for the edge",
                                    options=[node for node in node_list if node != node1_select],
                                    format_func=custom_format_func, key="node2_select")
        new_target = node2_select['id']

        # Check if the source and target nodes are the same
        if new_source == new_target:
            st.error("Source and target nodes cannot be the same.")
        else:
            new_label = st.text_input("Enter new label for the edge", value=selected_edge["label"])
            update_edge_button = st.button("Update Edge", key="update_edge_button", type="primary",
                                           use_container_width=True)

            if update_edge_button:
                # Update edge properties
                edge_list[selected_index]["source"] = new_source
                edge_list[selected_index]["target"] = new_target
                edge_list[selected_index]["label"] = new_label

                # Update session state with the modified edge list
                st.session_state["edge_list"] = edge_list

                st.success(f"Edge '{selected_edge_description}' has been updated.")
                st.rerun()

    except ValueError:
        st.error("There are no relations added yet. Please create relations between nodes or import a graph")


# Function to delete an edge
def delete_edge():
    edge_list = st.session_state["edge_list"]
    node_list = st.session_state["node_list"]
    edge_descriptions = []
    source_node_label = ''
    target_node_label = ''
    for edge in edge_list:
        for node in node_list:
            if node["id"] == edge["source"]:
                source_node_label = node['data']['label']
            elif node["id"] == edge["target"]:
                target_node_label = node['data']['label']
        edge_descriptions.append(f"{source_node_label} -> {target_node_label} ({edge['label']})")

    # Select edge to delete
    edge_to_delete = st.selectbox("Select edge to delete", key="edge_to_delete", options=edge_descriptions)
    delete_edge_button = st.button("Delete Edge", key="delete_edge_button", type="primary", use_container_width=True)

    if delete_edge_button:
        # Find the index of the selected edge in the options list
        selected_index = edge_descriptions.index(edge_to_delete)

        # Remove the selected edge from the edge list using its index
        del edge_list[selected_index]

        # Update the session state with the modified edge list
        st.session_state["edge_list"] = edge_list

        st.success(f"Edge '{edge_to_delete}' has been deleted.")
        time.sleep(1)
        st.experimental_rerun()


# Function to display stored graph details
def store_graph():
    with st.expander("Show Individual Lists"):
        st.json(st.session_state["node_list"], expanded=False)
        st.json(st.session_state["edge_list"], expanded=False)

    with st.expander("Show Graph JSON", expanded=False):
        update_graph_dict()
        st.json(st.session_state["graph_dict"])


# Function to visualize graph using Graphviz and Agraph
def visualize_graph():
    update_graph_dict()

    def set_color(node_type_for_color):
        color = 'grey'
        if node_type_for_color == 'product':
            color = 'red'
        elif node_type_for_color == 'process':
            color = 'green'
        elif node_type_for_color == 'connector':
            color = 'blue'
        return color

    mechanical_view_graph = None
    engineer_view_graph = None
    sustainability_view_graph = None

    if st.session_state['view_graphs']:
        for graph in st.session_state['view_graphs']:
            if "Mechanical View Graph" in graph:
                mechanical_view_graph = graph["Mechanical View Graph"]
            elif "Basic Engineering View Graph" in graph:
                engineer_view_graph = graph["Basic Engineering View Graph"]
            elif "Sustainability View Graph" in graph:
                sustainability_view_graph = graph["Sustainability View Graph"]
    selected_graph_to_view = st.selectbox("Select the Graph to view",
                                          options=["Main Graph", "Mechanical View Graph",
                                                   "Basic Engineering View Graph", "Sustainability View Graph"])

    if selected_graph_to_view == 'Main Graph':
        graph_dict = st.session_state["graph_dict"]
        node_list = graph_dict["nodes"]
        edge_list = graph_dict["edges"]
    elif selected_graph_to_view == 'Mechanical View Graph' and mechanical_view_graph:
        graph_dict = mechanical_view_graph
        node_list = graph_dict["nodes"]
        edge_list = graph_dict["edges"]
    elif selected_graph_to_view == 'Basic Engineering View Graph' and engineer_view_graph:
        graph_dict = engineer_view_graph
        node_list = graph_dict["nodes"]
        edge_list = graph_dict["edges"]
    elif selected_graph_to_view == 'Sustainability View Graph' and sustainability_view_graph:
        graph_dict = sustainability_view_graph
        node_list = graph_dict["nodes"]
        edge_list = graph_dict["edges"]
    elements = []

    try:
        with st.expander("Graph Visualization with Cytoscape", expanded=True):
            for node in node_list:
                elements.append(
                    {
                        "data": {
                            "id": node['id'],
                            "data": node['data'],
                            "label": node['data']['label']
                        }
                    }
                )

            for edge in edge_list:
                elements.append(
                    {
                        "data": {
                            "source": edge['source'],
                            "target": edge['target'],
                            "id": edge['id'],
                        },
                        "selectable": False,
                    }
                )

            stylesheet = [
                {"selector": "node", "style": {"label": "data(label)", "width": 20, "height": 20}},
                {
                    "selector": "edge",
                    "style": {
                        "width": 3,
                        "curve-style": "bezier",
                        "target-arrow-shape": "triangle",
                    },
                },
            ]

            # Visualize the graph
            cytoscape_graph_col, property_table_col = st.columns([0.6, 0.4])
            with cytoscape_graph_col:
                selected = cytoscape(
                    width='900px',
                    height="500px",
                    elements=elements,
                    stylesheet=stylesheet,
                    key='graph',
                    selection_type='single',
                    min_zoom=0.7,
                    max_zoom=2
                )
            with property_table_col:

                if len(selected['nodes']) > 0:
                    views = []
                    for node in node_list:
                        if selected['nodes'][0] == node['id']:
                            views = node['data']['props']['views']

                    view_names = []
                    if len(views) > 0:
                        view_names = list(views.keys())
                    else:
                        st.warning("No views available for selected node")

                    selected_view = st.selectbox('Select View', view_names)
                    selected_view_properties = []
                    if selected_view:
                        for view_name, view_data in views.items():
                            if view_name == selected_view:
                                selected_view_properties = view_data['properties']

                        # Create a DataFrame to display in the table
                        st.subheader(f"Properties of {selected_view} ")

                        if selected_view_properties is not None:
                            parameter_keys = ['name', 'target_value', 'min_value', 'max_value', 'unit']
                            table_data = []
                            for item in selected_view_properties:
                                row = {}
                                for key in parameter_keys:
                                    if key in item:
                                        row[key.capitalize()] = item[key]
                                table_data.append(row)
                            dataframe = pd.DataFrame(table_data)
                            st.dataframe(dataframe, hide_index=True, column_config={
                                "Name": "Name",
                                "Target_value": "Target Value",
                                "Min_value": "Min. Value",
                                "Max_value": "Max. Value",
                                "Unit": "Unit"
                            })
                else:
                    st.info("Please click on any of the node to see the views and properties of it")
        with st.expander("Graph Visualization with Graphviz", expanded=False):
            graph = graphviz.Digraph()
            for node in node_list:
                node_id = node["id"]
                node_label = node["data"]["label"] if "data" in node and "label" in node["data"] else ""
                node_type = node["type"]
                graph.node(node_id, label=node_label, color=set_color(node_type))

            for edge in edge_list:
                source = edge["source"]
                target = edge["target"]
                label = edge["label"]
                graph.edge(source, target, label)

            st.graphviz_chart(graph)
        with st.expander("Graph Visualization with Agraph", expanded=False):
            nodes = []
            edges = []

            for node in node_list:
                node_id = node["id"]
                node_label = node["data"]["label"] if "data" in node and "label" in node["data"] else ""
                nodes.append(Node(id=node_id, label=node_label))

            for edge in edge_list:
                source = edge["source"]
                target = edge["target"]
                label = edge["label"]
                edges.append(Edge(source=source, target=target, label=label))

            config = Config(width=1000,
                            directed=True,
                            physics=True,
                            heirarchical=False,
                            nodeHighlightBehavior=True,
                            highlightColor="#F7A7A6",
                            collapsible=False,

                            )

            agraph(nodes=nodes,
                   edges=edges,
                   config=config)
    except:
        st.error("Selected view graph is not available or empty. ")


# Function to analyze graph
def analyze_graph():
    update_graph_dict()
    g = nx.DiGraph()
    graph_dict = st.session_state["graph_dict"]
    node_list = graph_dict["nodes"]
    edge_list = graph_dict["edges"]
    node_tuple_list = []
    edge_tuple_list = []

    for node in node_list:
        node_tuple = (node["id"], node)  # Change "name" to "id"
        node_tuple_list.append(node_tuple)

    for edge in edge_list:
        edge_tuple = (edge["source"], edge["target"], edge)  # Change "source" and "target" keys
        edge_tuple_list.append(edge_tuple)

    g.add_nodes_from(node_tuple_list)
    g.add_edges_from(edge_tuple_list)

    select_function = st.selectbox(label="Select function",
                                   options=["Output nodes and edges", 'Count nodes', "Show specific node",
                                            "Show specific edge", "Check Path", "Check if Graph is Empty",
                                            "Density of Graph", "Is Graph Directed", "Find shortest Path",
                                            "Show shortest Path(Soln of Prof.Luder)", "Spanning Tree",
                                            "Minimum Spanning Tree", "Recurring"])

    if select_function == "Output nodes and edges":
        output_nodes_and_edges(graph=g)
    elif select_function == "Count nodes":
        count_nodes(graph=g)
    elif select_function == "Check Path":
        check_path(graph=g)
    elif select_function == "Check if Graph is Empty":
        is_empty(graph=g)
    elif select_function == "Density of Graph":
        find_density(graph=g)
    elif select_function == "Is Graph Directed":
        is_directed(graph=g)
    elif select_function == "Show specific node":
        specific_node(graph=g)
    elif select_function == "Show specific edge":
        specific_edge(graph=g)
    elif select_function == "Find shortest Path":
        shortest_path(g)
    # elif select_function == "Show shortest Path(Soln of Prof.Luder)":
    #     show_shortest_paths(g)
    elif select_function == "Spanning Tree":
        spanning_tree(g)
    elif select_function == "Minimum Spanning Tree":
        minimum_spanning_tree(g)
    elif select_function == "Recurring":
        spectral_clustering(g, 5)


def system_analysis_function():
    update_graph_dict()
    g = nx.DiGraph()
    graph_dict = st.session_state["graph_dict"]
    node_list = graph_dict["nodes"]
    edge_list = graph_dict["edges"]
    node_tuple_list = []
    edge_tuple_list = []

    for node in node_list:
        node_tuple = (node["id"], node)  # Change "name" to "id"
        node_tuple_list.append(node_tuple)

    for edge in edge_list:
        edge_tuple = (edge["source"], edge["target"], edge)  # Change "source" and "target" keys
        edge_tuple_list.append(edge_tuple)

    g.add_nodes_from(node_tuple_list)
    g.add_edges_from(edge_tuple_list)

    system_analysis(g)


# Function to export graph to JSON
def export_graph():
    visualize_graph()
    graph_string = json.dumps(st.session_state["graph_dict"])

    st.download_button("Export Graph to JSON",
                       file_name="graph.json",
                       mime="application/json",
                       data=graph_string,
                       use_container_width=True,
                       type="primary"
                       )
