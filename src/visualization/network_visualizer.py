# src/visualization/network_visualizer.py

import json

class NetworkVisualizer:
    def __init__(self, bayesian_network):
        self.bayesian_network = bayesian_network

    def generate_graph_data(self):
        nodes = []
        links = []

        for node_id, node in self.bayesian_network.nodes.items():
            nodes.append({"id": node_id, "name": node.name})

        for edge in self.bayesian_network.edges:
            links.append({"source": edge.parent.id, "target": edge.child.id})

        return {"nodes": nodes, "links": links}

    def generate_html(self):
        graph_data = self.generate_graph_data()
        return json.dumps(graph_data)