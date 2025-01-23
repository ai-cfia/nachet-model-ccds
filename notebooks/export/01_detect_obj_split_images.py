#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import requests
import cv2
from dotenv import load_dotenv

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


# In[ ]:


# Function to save individual seed images
def save_cropped_seeds(image_path, seeds_bboxes, output_folder, file_prefix):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    for idx, bbox in enumerate(seeds_bboxes):
        x_min, y_min, x_max, y_max = bbox
        cropped_image = image[y_min:y_max, x_min:x_max]
        cropped_filename = f"{file_prefix}_seed{idx}.tiff"
        cropped_filepath = os.path.join(output_folder, cropped_filename)
        cv2.imwrite(cropped_filepath, cropped_image)


# In[ ]:


# Function to call the model REST endpoint
def get_seed_detections(image_path):
    with open(image_path, "rb") as image_file:
        response = requests.post(
            url=SEED_DETECTOR_URL,  # Replace with your REST endpoint URL
            files={"file": (os.path.basename(image_path), image_file)},
            headers={"Content-Type": "multipart/form-data"},
        )
        response.raise_for_status()
        return response.json()["predictions"]


# In[ ]:


def process_folder(folder_path, output_folder, logpath):
    for item in os.listdir(folder_path):
        print(item)
        item_path = os.path.join(folder_path, item)
        if os.path.isdir(item_path):
            output_sub_folder = os.path.join(output_folder, item)
            process_folder(item_path, output_sub_folder)
        else:
            if item.endswith(IMAGE_FORMAT):
                image_path = item_path
                try:
                    seed_detections = get_seed_detections(image_path)
                    if seed_detections:
                        save_cropped_seeds(
                            image_path,
                            seed_detections,
                            output_folder,
                            image_path.split(".")[0],
                        )
                    else:
                        # error_log.append(image_path)
                        raise Exception("Error processing image " + image_path)
                except Exception as e:
                    with open(logpath, "a") as error_file:
                        error_file.write(str(e) + "\n")

def main():
    process_folder(DIR_RAW_IMAGES, DIR_OUTPUT_IMAGES, "error_log.txt")

main()


# In[ ]:


# # Main script
# def main(data_folder, output_folder):
#     error_log = []
#     for label_folder in os.listdir(data_folder):
#         print(label_folder)
#         label_folder_path = os.path.join(data_folder, label_folder)
#         if os.path.isdir(label_folder_path):
#             for image_file in os.listdir(label_folder_path):
#                 if image_file.endswith(IMAGE_FORMAT):
#                     image_path = os.path.join(label_folder_path, image_file)
#                     try:
#                         seed_detections = get_seed_detections(image_path)
#                         if seed_detections:
#                             save_cropped_seeds(
#                                 image_path,
#                                 seed_detections,
#                                 output_folder,
#                                 image_file.split(".")[0],
#                             )
#                         else:
#                             error_log.append(image_path)
#                     except requests.RequestException as e:
#                         error_log.append(image_path)
#                         print(f"Request failed for {image_path}: {e}")

#     # Write error log
#     with open("error.txt", "w") as error_file:
#         for error_image in error_log:
#             error_file.write(error_image + "\n")

# main(DIR_RAW_IMAGES, DIR_OUTPUT_IMAGES)

# Replace 'data_folder' and 'output_folder' with your actual folders


# In[ ]:


# def main(source, subfolders, dest):
#     error_log = []
#     subfolder_combinations = []
#     for i in range(len(subfolders[0])):
#         for j in range(len(subfolders[1])):
#             for k in range(len(subfolders[2])):
#                 subfolder_combinations.append( source + "/" + subfolders[0][i] + "/" +subfolders[1][j] + "/" +subfolders[2][k])

#     # Write error log
#     with open("error.txt", "w") as error_file:
#         for error_image in error_log:
#             error_file.write(error_image + "\n")

# main(
#     DIR_RAW_IMAGES, [DIR_ZOOM_LVL, ["Training", "Testing"], SPECIES], DIR_OUTPUT_IMAGES
# )


# In[ ]:




