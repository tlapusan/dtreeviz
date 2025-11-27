import datetime
import json
import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import SystemMessage, HumanMessage

from dtreeviz.models.shadow_decision_tree import ShadowDecTree
from dtreeviz.models.sklearn_decision_trees import ShadowSKDTree

_ = load_dotenv(find_dotenv())  # read local .env file

DEFAULT_LLM_MODEL = "gpt-4.1-mini"

# Store for chat message histories (session-based)
_chat_history_store = {}

leaf_template_string = """You are an expert in decision tree models. You have very good knowledge about tree structure interpretation. 
                        You should give advices how the decision tree could be improved based on its current structure.
                        
                        I have the leaf nodes of a decision tree in a json format, delimited by '''.
                        I would like you to tell me about the following:
                        {leaf_related_questions}
                        json format: ```{leaf_json_format}```
                        
                        At the end of your response, simple output the following text " Ask me more by typing viz_model.question('Your question')"
                    """

node_stats_template_string = """
    Bellow I will provide you, in json format, some basic statistics for numeric and string columns in a json format delimited by '''.
    Those statistics are for the node samples and for the training set samples.
    I will also provide you the distribution of class labels for that specific node as follow : {target_stats}
    Start with a very short description of the node class labels.
    After that, please make a short analysis only for the features which you consider have the most contribution on the node prediction, 
    based on the stats from node vs training samples. Please take in consideration all the stats provided, like count, mean, std (stadard deviation), min, max, 25% (percentile 25), 50% (percentile 50) and 75% (percentile 75)
    At the end, make a summary of the all feature stats, with their implications in the node label stats. 
    
    node sample statistics : '''{node_stats_json_format}'''
    training sample statistics : '''{training_stats_json_format}'''  
    """

tree_stats_template_string = """
    Bellow I will provide you, in json format, more information about the tree structure, leaf and internal nodes stats and training set statistics, all delimited by '''.
        
    tree structure information: '''{tree_structure_knowledge}'''
    leaf nodes information: '''{leaf_nodes_knowledge}'''
    internal nodes information: '''{internal_nodes_knowledge}'''
    training set information: '''{training_set_knowledge}'''
    """

leaf_prompt_template = ChatPromptTemplate.from_template(leaf_template_string)
node_stats_template = ChatPromptTemplate.from_template(node_stats_template_string)
tree_stats_template = ChatPromptTemplate.from_template(tree_stats_template_string)


def _convert_to_json_serializable(obj):
    """Convert NumPy types and other non-JSON-serializable objects to native Python types.
    Compatible with both NumPy 1.x and 2.x.
    """
    # Handle NumPy integers (compatible with NumPy 2.0)
    # np.integer is the abstract base class that works in both NumPy 1.x and 2.x
    if isinstance(obj, np.integer):
        return int(obj)
    # Handle NumPy floating point numbers (compatible with NumPy 2.0)
    # np.floating is the abstract base class that works in both NumPy 1.x and 2.x
    elif isinstance(obj, np.floating):
        return float(obj)
    # Handle NumPy arrays
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    # Handle NumPy booleans
    elif isinstance(obj, np.bool_):
        return bool(obj)
    # Handle dictionaries recursively
    elif isinstance(obj, dict):
        return {key: _convert_to_json_serializable(value) for key, value in obj.items()}
    # Handle lists and tuples recursively
    elif isinstance(obj, (list, tuple)):
        return [_convert_to_json_serializable(item) for item in obj]
    # Handle pandas NaN values
    elif pd.isna(obj):
        return None
    else:
        return obj


def get_completion(prompt, model=None):
    """Get a completion from the LLM using the new ChatOpenAI interface.
    
    Args:
        prompt: The prompt text to send to the LLM
        model: OpenAI model to use. If None, defaults to DEFAULT_LLM_MODEL.
    """
    if model is None:
        model = DEFAULT_LLM_MODEL
    chat = ChatOpenAI(temperature=0, model=model)
    response = chat.invoke([HumanMessage(content=prompt)])
    return response.content


def _get_library(tree: ShadowDecTree):
    if isinstance(tree, ShadowSKDTree):
        return "Scikit-Learn"
    return None


def _get_tree_structure_knowledge(tree: ShadowDecTree):
    tree_structure_knowledge = {}

    tree_structure_knowledge["tree type"] = "classification" if tree.is_classifier() else "regression"
    try:
        tree_structure_knowledge["criterion"] = tree.criterion()
    except Exception as e:
        pass

    try:
        tree_structure_knowledge["tree max depth"] = tree.get_max_depth()
    except Exception as e:
        pass
    tree_structure_knowledge["number of nodes"] = tree.nnodes()
    return tree_structure_knowledge


def _get_training_set_knowledge(tree: ShadowDecTree):
    training_set_knowledge = {}
    training_df = pd.DataFrame(tree.X_train, columns=tree.feature_names).convert_dtypes()
    training_target_df = pd.Series(tree.y_train)
    training_set_stats = training_df.describe(include='all').to_json()

    training_set_knowledge["training set size"] = training_df.shape[0]
    training_set_knowledge["feature list"] = tree.feature_names
    training_set_knowledge["feature list size"] = len(tree.feature_names)
    training_set_knowledge["feature summary stats"] = training_set_stats

    training_set_knowledge["target class"] = tree.target_name
    training_set_knowledge["target class number"] = tree.nclasses()
    training_set_knowledge["target class distribution"] = training_target_df.value_counts().to_json()

    return training_set_knowledge


def _get_leaf_nodes_knowledge(tree: ShadowDecTree):
    leaf_nodes_knowledge = {}
    for leaf in tree.leaves:
        leaf_info = {
            "data samples": leaf.nsamples(),
            "prediction": leaf.prediction(),
            "leaf level": leaf.level,
            "node criterion": tree.criterion()
        }

        if tree.is_classifier():
            class_counts = tree.get_node_nsamples_by_class(leaf.id)
            node_nsamples_by_class = [
                f"class label {i} contains {int(node_sample_count)} samples"
                for i, node_sample_count in enumerate(class_counts)
            ]
            total_samples = sum(class_counts) if class_counts is not None else 0
            prediction_confidence = round(
                max(class_counts) / total_samples, 2) if class_counts is not None and total_samples else None

            leaf_info.update({
                "prediction class": leaf.prediction_name(),
                "prediction confidence": prediction_confidence,
                "leaf sample counts": ", ".join(node_nsamples_by_class)
            })
        else:
            samples = leaf.samples()
            if len(samples):
                y_values = tree.y_train[samples]
                leaf_info.update({
                    "prediction mean": round(float(np.mean(y_values)), 4),
                    "prediction std": round(float(np.std(y_values)), 4),
                    "prediction min": round(float(np.min(y_values)), 4),
                    "prediction max": round(float(np.max(y_values)), 4)
                })
        leaf_nodes_knowledge[f"leaf node id {leaf.id}"] = leaf_info
    return leaf_nodes_knowledge


def _get_internal_nodes_knowledge(tree: ShadowDecTree):
    internal_nodes_knowledge = {}
    for node in tree.internal:
        leaf_info = {
            "data samples": node.nsamples(),
            "categorical split": node.is_categorical_split(),
            "split threshold": node.split(),
            "split feature": node.feature_name(),
            "node purity": node.criterion(),
            "node level": node.level,
            "node criterion": tree.criterion()
        }

        if tree.is_classifier():
            node_nsamples_by_class = [
                f"class label {i} contains {int(node_sample_count)} samples"
                for i, node_sample_count in enumerate(tree.get_node_nsamples_by_class(node.id))
            ]
            leaf_info["leaf sample counts"] = ", ".join(node_nsamples_by_class)
        else:
            node_samples = node.samples()
            if len(node_samples):
                node_values = tree.y_train[node_samples]
                leaf_info.update({
                    "node target mean": round(float(np.mean(node_values)), 4),
                    "node target std": round(float(np.std(node_values)), 4),
                    "node target min": round(float(np.min(node_values)), 4),
                    "node target max": round(float(np.max(node_values)), 4)
                })
        internal_nodes_knowledge[f"internal node id {node.id}"] = leaf_info
    return internal_nodes_knowledge


def _get_session_history(session_id: str, max_messages: int = None) -> InMemoryChatMessageHistory:
    """Get or create a chat message history for a given session.
    
    Args:
        session_id: Unique identifier for the conversation session
        max_messages: Maximum number of messages to keep in history (excluding system message).
                     If None, no limit is applied. Old messages are trimmed when limit is exceeded.
    """
    if session_id not in _chat_history_store:
        _chat_history_store[session_id] = InMemoryChatMessageHistory()
    
    history = _chat_history_store[session_id]
    
    # Trim history if max_messages is set
    if max_messages is not None and max_messages > 0:
        messages = history.messages
        # Keep system message(s) and the most recent max_messages conversation messages
        # System messages are typically at the beginning
        from langchain_core.messages import SystemMessage
        system_messages = [msg for msg in messages if isinstance(msg, SystemMessage)]
        conversation_messages = [msg for msg in messages if not isinstance(msg, SystemMessage)]
        
        # Keep only the most recent max_messages conversation messages
        if len(conversation_messages) > max_messages:
            # Clear and rebuild with trimmed messages
            history.clear()
            # Re-add system messages first
            for msg in system_messages:
                history.add_message(msg)
            # Re-add only the most recent conversation messages
            for msg in conversation_messages[-max_messages:]:
                history.add_message(msg)
                    
    return history


def setup_chat(tree: ShadowDecTree, session_id: str = "default", model: str = None, max_history_messages: int = 20):
    """Setup a chat conversation with memory using LangChain Core.
    
    Args:
        tree: The shadow decision tree to analyze
        session_id: Unique identifier for the conversation session
        model: OpenAI model to use (e.g., "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo").
               If None, defaults to "gpt-4.1-mini" (good balance of cost and quality).
        max_history_messages: Maximum number of conversation messages to keep in history 
                              (excluding system message). Default is 20. Set to None for unlimited.
                              Old messages are automatically trimmed when limit is exceeded.
    """
    # Reset chat history store to start fresh for each setup
    global _chat_history_store
    _chat_history_store = {}
    
    # Use provided model or default
    if model is None:
        model = DEFAULT_LLM_MODEL
    ml_library = _get_library(tree)
    if ml_library is None:
        raise ValueError(
            "AI chat setup aborted: this tree's library is not currently supported. Only Scikit-Learn is supported for now."
        )
    leaf_nodes_knowledge = _get_leaf_nodes_knowledge(tree)
    internal_nodes_knowledge = _get_internal_nodes_knowledge(tree)
    training_set_knowledge = _get_training_set_knowledge(tree)
    tree_structure_knowledge = _get_tree_structure_knowledge(tree)

    # Format the tree statistics message
    # Convert NumPy types to native Python types for JSON serialization
    tree_stats_content = tree_stats_template.format_messages(
        tree_structure_knowledge=json.dumps(_convert_to_json_serializable(tree_structure_knowledge), indent=2),
        leaf_nodes_knowledge=json.dumps(_convert_to_json_serializable(leaf_nodes_knowledge), indent=2),
        internal_nodes_knowledge=json.dumps(_convert_to_json_serializable(internal_nodes_knowledge), indent=2),
        training_set_knowledge=json.dumps(_convert_to_json_serializable(training_set_knowledge), indent=2)
    )[0].content

    # Create comprehensive system prompt with all tree context
    # Use format() instead of f-string to avoid template parsing issues
    system_prompt_template = """You are an AI assistant specialized in Machine Learning, especially in decision tree structure interpretation.
You are not just talkative, you thrive on providing in-depth details and insights related to provided decision tree structure.
Always be helpful, detailed, and insightful when answering questions about decision trees.

Respond directly and naturally, as if you have direct knowledge of the tree. Avoid phrases like "based on the provided information", "from the provided data", "according to the information", or similar references. Simply state facts and insights directly.

The machine learning library used for this decision tree is: {ml_library}

Below is detailed information about this decision tree in JSON format:

{tree_stats_content}

Use this information to answer questions about the decision tree structure, nodes, leaves, training data, and any other aspects of the model."""
    
    # Format the system prompt with actual values
    # Escape curly braces in JSON so ChatPromptTemplate treats them as literals
    escaped_tree_stats = tree_stats_content.replace("{", "{{").replace("}", "}}")
    system_prompt = system_prompt_template.format(
        ml_library=ml_library,
        tree_stats_content=escaped_tree_stats
    )
    
    # Create the prompt template with system message
    # Since system_prompt is already a fully formatted string, we create it as a literal
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        ("placeholder", "{messages}")
    ])

    # Create the LLM
    # Note: streaming is controlled at the chat() method level, not here
    # We keep streaming=False here so invoke() works normally
    chat = ChatOpenAI(temperature=0.0, model=model)

    # Create the chain with message history
    # Use a lambda to pass max_history_messages to _get_session_history
    def get_session_history_with_limit(session_id: str) -> InMemoryChatMessageHistory:
        return _get_session_history(session_id, max_messages=max_history_messages)
    
    chain = prompt | chat
    conversation = RunnableWithMessageHistory(
        chain,
        get_session_history_with_limit,
        input_messages_key="messages"
    )

    # Initialize the conversation history with context
    config = {"configurable": {"session_id": session_id}}
    

    return conversation, config


# TOOD apply function calling ?
def build_node_stats_prompt(shadow_tree: ShadowDecTree, node_id: int) -> str:
    """Build a prompt for explaining node statistics for a specific node."""
    node_samples = shadow_tree.get_node_samples()
    df = pd.DataFrame(shadow_tree.X_train, columns=shadow_tree.feature_names).convert_dtypes()
    training_stats = df.describe(include='all').to_json()

    node_stats = df.iloc[node_samples[node_id]].describe(include='all').to_json()

    if shadow_tree.is_classifier():
        target_values = shadow_tree.y_train[node_samples[node_id]]
        unique_values, value_counts = np.unique(target_values, return_counts=True)
        target_frequencies = dict(zip(unique_values, value_counts))
        target_string = ""
        for key, value in target_frequencies.items():
            target_string += (
                f"Target class {shadow_tree.class_names[key] if shadow_tree.class_names is not None else key} "
                f"has {value} samples. "
            )
    else:
        target_values = shadow_tree.y_train[node_samples[node_id]]
        target_string = (
            f"The regression target statistics for this node are: "
            f"mean={float(np.mean(target_values)):.4f}, "
            f"std={float(np.std(target_values)):.4f}, "
            f"min={float(np.min(target_values)):.4f}, "
            f"max={float(np.max(target_values)):.4f}."
        )

    messages = node_stats_template.format_messages(
        node_stats_json_format=node_stats,
        target_stats=target_string,
        training_stats_json_format=training_stats
    )
    return messages[0].content

