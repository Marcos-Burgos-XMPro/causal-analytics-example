"""
Default script template for the Python Meta Action Agent.

When importing packages, follow the format below to add a comment at the end of declaration 
and specify a version or a package name when the import name is different from expected python package.
This allows the agent to install the correct package version during configuration:
e.g. import paho.mqtt as np  # version=2.1.0 package=paho-mqtt

This script provides a structure for implementing on_create, on_receive, and on_destroy functions.
It includes a basic example using 'foo' and 'bar' concepts to demonstrate functionality.
Each function should return a dictionary object with result data, or None if no result is needed.
"""

def on_create(data: dict) -> dict | None:
    return None

# Import necessary libraries
import networkx as nx
from dowhy import gcm
from datetime import datetime
import warnings
import json
import numpy as np
import pandas as pd
import ast

# Suppress all warnings
warnings.filterwarnings("ignore")

def on_receive(data: dict) -> dict:
    """
    Perform causal analysis on a specified target node using observational data and predefined causal relationships.

    This function reads observational engine data from a CSV file, constructs a causal graph based on 
    domain-specific relationships, and computes both direct arrow strengths and intrinsic causal influences 
    on the given target node. The results include confidence intervals and are formatted in both node and edge views.

    Args:
        data (dict): A dictionary containing:
            - "observation" (list[dict]): Observational data where each dictionary represents a data row 
              with keys as variable names and values as floats or strings (e.g., timestamps).
            - "causal_relationships" (str): A string representation of a list of tuples defining causal 
              edges between variables, e.g., [('altitude', 'engine_load'), ...].
            - "target_node" (str): Name of the variable to analyze as the effect node in causal queries.

    Returns:
        dict: A dictionary containing:
            - "timestamp" (str): Execution timestamp in 'YYYY-MM-DD HH:MM:SS' format.
            - "status" (str): "success" if computation succeeded, otherwise "error".
            - "message" (str): Success message or error details.
            - "target_node" (str): The node under causal evaluation.
            - "arrow_strength_edge" (str): JSON string mapping causal edges (source, target) to strength values.
            - "arrow_strengths_edge_intervals" (str): JSON string mapping edges to [lower, upper] confidence bounds.
            - "arrow_strength_node" (str): JSON string mapping treatment nodes to strength values.
            - "arrow_strengths_node_intervals" (str): JSON string mapping treatment nodes to confidence bounds.
            - "intrinsic_influence" (str): JSON string mapping treatment nodes to intrinsic influence scores.
            - "intrinsic_influence_intervals" (str): JSON string mapping treatment nodes to confidence intervals.
    """
    # Record the current timestamp for logging and traceability
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Ensure reproducibility by setting a fixed random seed
    gcm.util.general.set_random_seed(0)

    try:
        # === 1. MODEL SETUP ===
        # Step 1.1: Load observational data into a pandas DataFrame
        # Handle both JSON string (Meta Agent) and dictionary (Local Agent) formats
        if isinstance(data['observation'], str):
            deserialized_data = json.loads(data['observation'])
        else:
            deserialized_data = data['observation']
        observation = pd.DataFrame(deserialized_data)

        # Step 1.2: Parse the causal relationships and define the graph structure
        causal_relationship = ast.literal_eval(data["causal_relationships"].strip())
        causal_graph = nx.DiGraph(causal_relationship)
        causal_model = gcm.InvertibleStructuralCausalModel(causal_graph)

        # === 2. AUTOMATIC MECHANISM ASSIGNMENT ===
        # Extract target node from input data
        target_node = data.get("target_node")
        # Assign default generative models (causal mechanisms) to each node
        summary_auto_assignment = gcm.auto.assign_causal_mechanisms(causal_model, observation)

        # === 3. DIRECT CAUSAL STRENGTH (ARROW STRENGTH) ===
        # Estimate the strength of direct causal links to the target node using bootstrapped intervals
        arrow_strengths_median, arrow_strengths_intervals = gcm.confidence_intervals(
            gcm.fit_and_compute(
                gcm.arrow_strength,
                causal_model,
                bootstrap_training_data=observation,
                target_node=target_node
            )
        )
        arrow_strengths = arrow_strengths_median

        # Format results using node-level notation: {treatment: strength}
        arrow_strengths_node = dict(
            sorted(
                ((treatment, round(value, 2)) for (treatment, _), value in arrow_strengths.items()),
                key=lambda item: item[1],
                reverse=True
            )
        )
        arrow_strengths_intervals_node = {
            treatment: [round(x, 2) for x in value.tolist()]
            for (treatment, _), value in arrow_strengths_intervals.items()
        }

        # Format results using edge-level notation: {"(source, target)": strength}
        arrow_strengths_edge_str = {
            f"({k[0]}, {k[1]})": round(v, 2) for k, v in arrow_strengths.items()
        }
        arrow_strengths_intervals_edge_str = {
            f"({k[0]}, {k[1]})": [round(x, 2) for x in v.tolist()]
            for k, v in arrow_strengths_intervals.items()
        }

        # === 4. INTRINSIC CAUSAL INFLUENCE ===
        # Estimate intrinsic influence of each treatment on the target using randomized interventions
        intrinsic_influence_median, intrinsic_influence_intervals = gcm.confidence_intervals(
            gcm.fit_and_compute(
                gcm.intrinsic_causal_influence,
                causal_model,
                bootstrap_training_data=observation,
                target_node=target_node,
                num_samples_randomization=10
            )
        )
        intrinsic_influence = intrinsic_influence_median

        # Format results: sorted list of treatment influences on the target
        intrinsic_influence_dict = dict(
            sorted(
                ((treatment, round(value, 2)) for treatment, value in intrinsic_influence.items()),
                key=lambda item: item[1],
                reverse=True
            )
        )
        intrinsic_influence_intervals_dict = {
            treatment: [round(x, 2) for x in value.tolist()]
            for treatment, value in intrinsic_influence_intervals.items()
        }

        # === 5. RETURN SUCCESSFUL RESPONSE ===
        result = {
            "timestamp": timestamp,
            "status": "success",
            "message": "Causal query calculated successfully.",
            "target_node": target_node,
            "arrow_strength_edge": json.dumps(arrow_strengths_edge_str),
            "arrow_strengths_edge_intervals": json.dumps(arrow_strengths_intervals_edge_str),
            "arrow_strength_node": json.dumps(arrow_strengths_node),
            "arrow_strengths_node_intervals": json.dumps(arrow_strengths_intervals_node),
            "intrinsic_influence": json.dumps(intrinsic_influence_dict),
            "intrinsic_influence_intervals": json.dumps(intrinsic_influence_intervals_dict)
        }

    except Exception as e:
        # Handle any exception and return an error response with placeholders
        result = {
            "timestamp": timestamp,
            "status": "error",
            "message": str(e),
            "target_node": data.get("target_node", ""),
            "arrow_strength_edge": None,
            "arrow_strengths_edge_intervals": None,
            "arrow_strength_node": None,
            "arrow_strengths_node_intervals": None,
            "intrinsic_influence": None,
            "intrinsic_influence_intervals": None
        }

    return result

def on_destroy() -> dict | None:
    return None