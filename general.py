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
    Executes causal inference analysis based on the input query type ('planning' or 'anomaly') 
    using a provided causal graph and observational data.

    This function performs the following:
    - Parses observational data and a causal graph from the input dictionary.
    - Initializes a causal model using the DoWhy GCM library.
    - Depending on the query type:
        - For 'anomaly': computes causal attribution scores to identify likely root causes.
        - For 'planning': estimates causal arrow strengths and intrinsic influence of each variable on a specified target node.

    Args:
        data (dict): A dictionary containing the following keys:
            - "query_type" (str): Type of causal query; one of ["planning", "anomaly"].
            - "observation" (Union[str, list[dict]]): Observational dataset in either JSON string or list-of-dictionaries format.
            - "causal_relationships" (str): String representation of a list of directed edges defining the causal graph.
            - "target_node" (str): Name of the variable to analyze.
            - "anomaly_data" (str): JSON string of anomaly values (only required for 'anomaly' queries).

    Returns:
        dict: A dictionary containing the query results, with the following structure:
            - "timestamp": Execution timestamp.
            - "status": 'success' or 'error'.
            - "message": Result message or error explanation.
            - "target_node": The analyzed node.
            - If 'anomaly' query:
                - "anomaly_data": Input anomaly data.
                - "anomaly_attribution": Attribution scores (JSON-encoded).
                - "anomaly_attribution_confidence": Confidence intervals (JSON-encoded).
            - If 'planning' query:
                - "arrow_strength_edge": Direct causal strengths between pairs (JSON-encoded).
                - "arrow_strengths_edge_intervals": Confidence intervals for edge strengths.
                - "arrow_strength_node": Strength of incoming effects per variable (JSON-encoded).
                - "arrow_strengths_node_intervals": Confidence intervals for node strengths.
                - "intrinsic_influence": Estimated influence of each variable on the target.
                - "intrinsic_influence_intervals": Confidence intervals for intrinsic influence.

    Raises:
        Returns an error message and null results in the response if any part of the pipeline fails.

    Example:
        >>> result = on_receive({
                "query_type": "planning",
                "observation": [{"altitude": 100, "engine_rpm": 1500, ...}, ...],
                "causal_relationships": "[('altitude', 'air_filter_pressure'), ...]",
                "target_node": "egt_turbo_inlet",
                "anomaly_data": '{"altitude": [1], "ambient_temp": [2], ...}'
            })
    """
    # Record the current timestamp for audit and logging
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Set a fixed seed to ensure reproducibility of causal estimates
    gcm.util.general.set_random_seed(0)

    # Input data
    query_type = data.get("query_type", None)
    target_node = data.get("target_node", None)
    raw_anomaly_data = data.get("anomaly_data", None)
    raw_causal_relationships = data.get("causal_relationships", None)

    # Output variables
    message = None
    # For anomaly
    result_attribution_scores_dict = None
    result_attribution_scores_intervals_dict = None
    # For planning
    result_arrow_strengths_edge = None
    result_arrow_strengths_intervals_edge = None
    result_arrow_strengths_node = None
    result_arrow_strengths_intervals_node = None
    result_intrinsic_influence_dict = None
    result_intrinsic_influence_intervals_dict = None

    try:
        # === 1. MODEL SETUP ===
        # Step 1.1: Convert observational input into a DataFrame
        # Handles both JSON strings (Meta Agent) and already-parsed lists (Local)
        if isinstance(data['observation'], str):
            deserialized_data = json.loads(data['observation'])
        else:
            deserialized_data = data['observation']
        observation = pd.DataFrame(deserialized_data)

        # Step 1.2: Parse causal relationships into a directed graph and initialize causal model
        causal_graph = nx.DiGraph(ast.literal_eval(raw_causal_relationships.strip()))
        causal_model = gcm.InvertibleStructuralCausalModel(causal_graph)

        # === START Query Selection ===

        if query_type == "anomaly":
            # Step 1.3: Load anomaly data and anomalous node to analyze
            anomalous_node = target_node
            anomaly_data = pd.DataFrame(data=json.loads(raw_anomaly_data))

            # === 2. CAUSAL MECHANISM ASSIGNMENT ===
            # Assign default generative models to each node using the observational data
            summary_auto_assignment = gcm.auto.assign_causal_mechanisms(causal_model, observation)

            # === 3. ANOMALY ATTRIBUTION ===
            # Compute the causal attribution scores for the anomaly using bootstrapped confidence intervals
            attribution_scores_median, attribution_scores_intervals = gcm.confidence_intervals(
                gcm.fit_and_compute(
                    gcm.attribute_anomalies,
                    causal_model,
                    bootstrap_training_data=observation,
                    target_node=anomalous_node,
                    anomaly_samples=anomaly_data
                )
            )
            attribution_scores = attribution_scores_median

            # Format results: sorted attribution values and corresponding confidence intervals
            attribution_scores_dict = dict(
                sorted(
                    ((treatment, round(value, 2)) for treatment, value in attribution_scores.items()),
                    key=lambda item: item[1],
                    reverse=True
                )
            )
            attribution_scores_intervals_dict = {
                treatment: [round(x, 2) for x in value.tolist()]
                for treatment, value in attribution_scores_intervals.items()
            }
        
            result_attribution_scores_dict = json.dumps(attribution_scores_dict)
            result_attribution_scores_intervals_dict = json.dumps(attribution_scores_intervals_dict)
            message = "Successful Calculation of Anomaly Attribution"
        
        elif query_type == "planning":
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

            result_arrow_strengths_edge = json.dumps(arrow_strengths_edge_str)
            result_arrow_strengths_intervals_edge = json.dumps(arrow_strengths_intervals_edge_str)
            result_arrow_strengths_node = json.dumps(arrow_strengths_node)
            result_arrow_strengths_intervals_node = json.dumps(arrow_strengths_intervals_node)
            result_intrinsic_influence_dict = json.dumps(intrinsic_influence_dict)
            result_intrinsic_influence_intervals_dict = json.dumps(intrinsic_influence_intervals_dict)
            message = "Successful Calculation of Arrow Strength and Intrinsic Influence"

        # === END Selection ===

        # === 4. RETURN SUCCESSFUL RESPONSE ===
        result = {
            "timestamp": timestamp,
            "status": "success",
            "message": message,
            "query_type": query_type,
            "causal_relationships": raw_causal_relationships,
            "target_node": target_node,
            # For anomaly
            "anomaly_data": raw_anomaly_data,
            "anomaly_attribution": result_attribution_scores_dict,
            "anomaly_attribution_confidence": result_attribution_scores_intervals_dict,
            # For planning
            "arrow_strength_edge": result_arrow_strengths_edge,
            "arrow_strengths_edge_intervals": result_arrow_strengths_intervals_edge,
            "arrow_strength_node": result_arrow_strengths_node,
            "arrow_strengths_node_intervals": result_arrow_strengths_intervals_node,
            "intrinsic_influence": result_intrinsic_influence_dict,
            "intrinsic_influence_intervals": result_intrinsic_influence_intervals_dict
        }

    except Exception as e:
        # Handle unexpected errors and return a failure result
        result = {
            "timestamp": timestamp,
            "status": "error",
            "message": str(e),
            "query_type": query_type,
            "causal_relationships": raw_causal_relationships,
            "target_node": target_node,
            # For anomaly
            "anomaly_data": raw_anomaly_data,
            "anomaly_attribution": None,
            "anomaly_attribution_confidence": None,
            # For planning
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