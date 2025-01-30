#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import requests
import cv2
import json
import base64
import uuid
from dotenv import load_dotenv
from urllib.request import Request, urlopen
from datetime import datetime
from PIL import Image
from tqdm.notebook import tqdm

# Load environment variables
load_dotenv(".env.secrets.15spp_zoom_level_validation_models")
load_dotenv(".env.config.15spp_zoom_level_validation_models")
SEED_DETECTOR_URL = os.getenv("SEED_DETECTOR_URL")

DIR_RAW_IMAGES = os.getenv("DIR_RAW_IMAGES")
DIR_OUTPUT_IMAGES = os.getenv("DIR_OUTPUT_IMAGES")

DIR_ZOOM_LVL = os.getenv("DIR_ZOOM_LVL")
DIR_TRAIN_TEST = os.getenv("DIR_TRAIN_TEST")
SPECIES = os.getenv("SPECIES")

IMAGE_FORMAT = os.getenv("IMAGE_FORMAT")
DETECTOR_API_KEY = os.getenv("DETECTOR_API_KEY")

RUN_LOG = os.getenv("RUN_LOG")
ERR_LOG = os.getenv("ERR_LOG")


# In[2]:


def run_log(message, init=False):
    os.makedirs(os.path.dirname(RUN_LOG), exist_ok=True)
    with open(RUN_LOG, ("w" if init else "a")) as f:
        f.write(f"{datetime.now()}: {message}\n")


# In[3]:


def err_log(message, init=False):
    os.makedirs(os.path.dirname(ERR_LOG), exist_ok=True)
    with open(ERR_LOG, ("w" if init else "a")) as f:
        f.write(f"{datetime.now()}: {message}\n")


# In[4]:


# Function to save individual seed images
def save_cropped_seeds_cv2(image_path, seeds_bboxes, output_folder, file_prefix):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    for idx, bbox in enumerate(seeds_bboxes):
        # print(bbox['box'])
        topX = bbox["box"]["topX"]
        topY = bbox["box"]["topY"]
        bottomX = bbox["box"]["bottomX"]
        bottomY = bbox["box"]["bottomY"]

        # print(topX, topY, bottomX, bottomY)
        x_min = int(topX * image.shape[1])
        x_max = int(bottomX * image.shape[1])
        y_min = int(topY * image.shape[0])
        y_max = int(bottomY * image.shape[0])

        # print(x_max, x_min, y_max, y_min)
        cropped_image = image[y_min:y_max, x_min:x_max]

        cropped_image

        cropped_filename = f"{file_prefix}_seed{idx}.tiff"
        cropped_filepath = os.path.join(output_folder, cropped_filename)

        os.makedirs(output_folder, exist_ok=True)
        print(f"Saving cropped image to {cropped_filepath}")
        cv2.imwrite(cropped_filepath, cropped_image)


# In[5]:


def save_cropped_seeds_pil(image_path, seeds_bboxes, output_folder, file_prefix):
    image = Image.open(image_path)
    for idx, bbox in enumerate(seeds_bboxes):
        # print(bbox['box'])
        topX = bbox["box"]["topX"]
        topY = bbox["box"]["topY"]
        bottomX = bbox["box"]["bottomX"]
        bottomY = bbox["box"]["bottomY"]

        # print(topX, topY, bottomX, bottomY)
        x_min = int(topX * image.size[0])
        x_max = int(bottomX * image.size[0])
        y_min = int(topY * image.size[1])
        y_max = int(bottomY * image.size[1])

        # print(x_max, x_min, y_max, y_min)
        cropped_image = image.crop((x_min, y_min, x_max, y_max))

        cropped_filename = f"{file_prefix}_seed{idx}.tiff"
        cropped_filepath = os.path.join(output_folder, cropped_filename)

        os.makedirs(output_folder, exist_ok=True)
        run_log(f"Cropped image : {cropped_filepath}")
        cropped_image.save(cropped_filepath)


# In[6]:


# Function to call the model REST endpoint


def get_seed_detections(image_path):
    with open(image_path, "rb") as image_file:
        image_string = base64.b64encode(image_file.read())

        data = {
            "input_data": {
                "columns": ["image"],
                "index": [0],
                "data": [image_string.decode("utf-8")],
            }
        }
        body = str.encode(json.dumps(data))

        headers = {
            "Content-Type": "application/json",
            "Authorization": ("Bearer " + DETECTOR_API_KEY),
        }
        req = Request(SEED_DETECTOR_URL, body, headers, method="POST")
        # req = Request("http://192.168.x.x:12380/score", body, headers, method="POST")
        response = urlopen(req)

        result = response.read()
        result_object = [json.loads(result.decode("utf8"))]
        return result_object[0]["boxes"]


# In[7]:


def process_folder(folder_path, output_folder, progress_bar):
    for item in os.listdir(folder_path):
        # print(item)
        item_path = os.path.join(folder_path, item)
        if os.path.isdir(item_path):
            output_sub_folder = os.path.join(output_folder, item)
            process_folder(item_path, output_sub_folder, progress_bar)
        else:
            if item.endswith(IMAGE_FORMAT):
                image_path = item_path
                # print("Processing : " + image_path)
                run_log(f"Processing : {image_path}")
                try:
                    seed_detections = get_seed_detections(image_path)
                    # print(seed_detections[0]['boxes'])
                    if seed_detections:
                        save_cropped_seeds_pil(
                            image_path,
                            seed_detections,
                            output_folder,
                            # image_path.split(".")[0],
                            uuid.uuid4(),
                        )
                    else:
                        # error_log.append(image_path)
                        raise Exception("Error processing image " + image_path)
                    progress_bar.update(1)
                    # run_log(f"Completed : {completed_files}/{total_files}")
                except Exception as e:
                    err_log(f"Error processing image {image_path} : {e}")
                    print(e)
                    exit(1)


def main():
    run_log("Starting seed detection process", init=True)
    err_log("Starting seed detection process", init=True)
    file_array = [len(files) for r, d, files in os.walk(DIR_RAW_IMAGES)]
    total_files = sum(file_array)
    progress_bar = tqdm(total=total_files)
    process_folder(DIR_RAW_IMAGES, DIR_OUTPUT_IMAGES, progress_bar)
    run_log("Seed detection process completed")
    err_log("Seed detection process completed")


main()

