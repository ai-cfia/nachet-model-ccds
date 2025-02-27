import json
import logging
import base64
import io

logger = logging.getLogger(__name__)

# Classes from primary model that should trigger secondary model
TRIGGER_CLASSES = ["class1", "class2", "class3"]

def preprocess(data, context):
    """
    Preprocess function for the workflow.
    
    Args:
        data: Input data received from the client
        context: Workflow context
        
    Returns:
        The preprocessed data to be passed to the primary model
    """
    if data is None:
        return None
    
    # For image data, we can pass it through directly
    # The model_handler.py will handle the image preprocessing
    return data

def check_trigger_class(data, context):
    """
    Check if the primary model output contains any trigger classes.
    Routes to either the secondary model or returns the primary result.
    
    Args:
        data: Output from the primary model
        context: Workflow context
        
    Returns:
        List with two elements: 
        1. Boolean indicating whether to route to secondary model
        2. The primary model results
    """
    try:
        # Parse model output if it's in string format
        if isinstance(data, str):
            predictions = json.loads(data)
        else:
            predictions = data
            
        # Check for trigger classes in the top prediction
        should_route_to_secondary = False
        
        if isinstance(predictions, list) and len(predictions) > 0:
            # Handle nested list structure (batch of results)
            if isinstance(predictions[0], list) and len(predictions[0]) > 0:
                top_class = predictions[0][0].get('class', '')
                if top_class in TRIGGER_CLASSES:
                    should_route_to_secondary = True
                    logger.info(f"Detected trigger class '{top_class}', routing to secondary model")
            # Handle single result structure
            elif isinstance(predictions[0], dict):
                top_class = predictions[0].get('class', '')
                if top_class in TRIGGER_CLASSES:
                    should_route_to_secondary = True
                    logger.info(f"Detected trigger class '{top_class}', routing to secondary model")
        
        return [should_route_to_secondary, data]
    except Exception as e:
        logger.error(f"Error in check_trigger_class: {str(e)}")
        # In case of error, don't route to secondary model
        return [False, data]

def return_primary_result(data, context):
    """
    Return the primary model result when secondary model is not needed.
    
    Args:
        data: List containing [should_route, primary_results]
        context: Workflow context
        
    Returns:
        The primary model results
    """
    # Extract only the primary results (second element in the list)
    return data[1]

def return_secondary_result(data, context):
    """
    Process and return the secondary model output.
    
    Args:
        data: Output from the secondary model
        context: Workflow context
        
    Returns:
        The secondary model results
    """
    # Secondary model output is used as is
    return data
