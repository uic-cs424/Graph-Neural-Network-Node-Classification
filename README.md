# Node-Classification-GNN

This repository contains code to implement and fit a Graph Neural Network (GNN) to our data. The multiclass classification is modeled as a node-level classification problem, where each table is represented as a directed graph, with each cell of the table corresponding to a node. The GNN model takes in the entire training dataset supergraph as input, and learns to predict each node's class through node embedding and edge connections data.

To run the experiment, follow the given steps - 
1. Install all the necessary libraries and dependencies in requirements.txt.
2. Upload the dataset as 'data.csv' to the root directory.
3. Run main.py.

All the files for visualization will be added to the root directory, examples from previously run experiments are given in the visualize folder.
