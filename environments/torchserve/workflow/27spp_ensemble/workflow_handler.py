import json
import logging
import base64
import io
from PIL import Image

logger = logging.getLogger(__name__)

def preprocess(data, context):
    """
    Preprocess function to handle image data and make it JSON serializable
    for the workflow.
    """
    logger.info(f"Workflow preprocess: data type: {type(data)}")
    
    try:
        # Special handling for curl -T uploads (raw HTTP body)
        if isinstance(data, list) and len(data) == 1 and isinstance(data[0], dict):
            if 'body' in data[0]:
                raw_bytes = data[0]['body']
                if isinstance(raw_bytes, (bytes, bytearray)):
                    logger.info(f"Found raw body data from curl upload, length: {len(raw_bytes)}")
                    # Pass through the raw bytes in a format the models can handle
                    encoded_string = base64.b64encode(raw_bytes).decode('utf-8')
                    return [{"image_data": encoded_string}]
                    
        # Process input based on type
        if isinstance(data, (bytes, bytearray)):
            # Log useful debugging info
            logger.info(f"Raw bytes data received, length: {len(data)}")
            
            # Verify this is actually image data by attempting to open it
            try:
                # Just test if this is valid image data
                Image.open(io.BytesIO(data)).verify()
                logger.info("Valid image data verified")
            except Exception as e:
                logger.warning(f"Not a valid image: {str(e)}")
                # Continue anyway, maybe the model will handle it
            
            # Convert bytes/bytearray to base64 string for JSON serialization
            encoded_string = base64.b64encode(data).decode('utf-8')
            processed_data = [{
                "image_data": encoded_string
            }]
            logger.info(f"Encoded data to base64 string of length: {len(encoded_string)}")
            return processed_data
            
        elif isinstance(data, list):
            logger.info(f"List data received, length: {len(data)}")
            processed_data = []
            
            for i, item in enumerate(data):
                if isinstance(item, (bytes, bytearray)):
                    logger.info(f"Item {i} is bytes of length: {len(item)}")
                    encoded_string = base64.b64encode(item).decode('utf-8')
                    processed_data.append({
                        "image_data": encoded_string
                    })
                elif isinstance(item, dict):
                    # Pass through dictionaries, but ensure values are JSON serializable
                    logger.info(f"Item {i} is dict with keys: {list(item.keys())}")
                    processed_item = {}
                    for k, v in item.items():
                        if isinstance(v, (bytes, bytearray)):
                            processed_item[k] = base64.b64encode(v).decode('utf-8')
                        else:
                            processed_item[k] = v
                    processed_data.append(processed_item)
                else:
                    # Log warning and skip non-supported types
                    logger.warning(f"Skipping item {i} with unsupported type: {type(item)}")
            
            logger.info(f"Preprocessing completed for {len(processed_data)} items")
            return processed_data
        
        else:
            logger.warning(f"Unsupported input data type: {type(data)}")
            # Return empty list to prevent workflow failure
            return []
        
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        # Return empty list to prevent workflow failure
        return []
        

def postprocess(data, context):
    """
    Post-process function to finalize workflow output
    """
    logger.info("Starting postprocessing")
    
    try:
        # Just pass through the data from the model
        logger.info("Postprocessing completed")
        return data
        
    except Exception as e:
        logger.error(f"Error in postprocessing: {str(e)}")
        return data
