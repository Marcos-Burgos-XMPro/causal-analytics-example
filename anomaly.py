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
    Evaluate causal attribution for observed anomalies in a specified node using a pre-trained causal model.

    This function receives observational and anomalous data, builds a causal model from predefined relationships,
    assigns causal mechanisms, and computes the contribution of each treatment node to the anomaly observed in a 
    target node. It returns both raw attribution scores and their associated confidence intervals.

    Args:
        data (dict): A dictionary containing:
            - "observation" (str | list[dict]): Either a JSON string or list of dictionaries representing 
              the observational dataset (rows of variable values).
            - "causal_relationships" (str): A string representation of a list of tuples defining causal 
              relationships, e.g., [('engine_load', 'engine_rpm'), ...].
            - "target_node" (str): The node used for causal mechanism fitting (typically the same as the anomalous node).
            - "anomalous_node" (str): The node in which an anomaly has been detected and attribution is to be performed.
            - "anomaly_data" (str): A JSON string representing the anomalous sample(s) as a list of records.

    Returns:
        dict: A dictionary containing:
            - "timestamp" (str): Time of execution.
            - "status" (str): "success" if execution completed, otherwise "error".
            - "message" (str): Message describing the outcome.
            - "anomalous_node" (str): The node analyzed for anomaly attribution.
            - "anomaly_data" (str): JSON string of the original anomaly data.
            - "anomaly_attribution" (str | None): JSON string mapping treatment nodes to their attribution scores, 
              sorted by importance.
            - "anomaly_attribution_confidence" (str | None): JSON string mapping treatment nodes to confidence intervals 
              [lower_bound, upper_bound] for the attribution scores.
    """
    # Record the current timestamp for audit and logging
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Set a fixed seed to ensure reproducibility of causal estimates
    gcm.util.general.set_random_seed(0)

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
        causal_relationship = ast.literal_eval(data["causal_relationships"].strip())
        causal_graph = nx.DiGraph(causal_relationship)
        causal_model = gcm.InvertibleStructuralCausalModel(causal_graph)

        # Step 1.3: Load anomaly data and anomalous node to analyze
        anomalous_node = data.get("anomalous_node")
        anomaly_data = pd.DataFrame(data=json.loads(data.get("anomaly_data")))

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

        # === 4. RETURN SUCCESSFUL RESPONSE ===
        result = {
            "timestamp": timestamp,
            "status": "success",
            "message": "Successful Calculation of Anomaly Attribution",
            "anomalous_node": data.get("anomalous_node"),
            "anomaly_data": json.dumps(data.get("anomaly_data")),
            "anomaly_attribution": json.dumps(attribution_scores_dict),
            "anomaly_attribution_confidence": json.dumps(attribution_scores_intervals_dict)
        }

    except Exception as e:
        # Handle unexpected errors and return a failure result
        result = {
            "timestamp": timestamp,
            "status": "error",
            "message": str(e),
            "anomalous_node": data.get("anomalous_node"),
            "anomaly_data": json.dumps(data.get("anomaly_data")),
            "anomaly_attribution": None,
            "anomaly_attribution_confidence": None
        }

    return result

def on_destroy() -> dict | None:
    return None