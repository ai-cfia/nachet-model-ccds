#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import json
import pandas as pd


# In[2]:


def load_classification_reports(root_path) -> dict:
    loaded_reports = {}
    for subdir, dirs, files in os.walk(root_path):
        for file_name in files:
            if file_name.endswith("_classification_report.json"):
                full_path = os.path.join(subdir, file_name)
                with open(full_path, "r") as f:
                    key = (file_name.split("_")[0]).split("-")[1]
                    loaded_reports[key] = json.load(f)
    return loaded_reports


# In[3]:


def train_state_to_df(train_state) -> pd.DataFrame:
    state_json = json.loads(train_state)
    result = []
    for log in state_json['log_history']:
        result.append({
            'epoch': log['epoch'],
            'grad_norm': log['grad_norm'],
            'loss': log['loss'],
            'learning_rate': log['learning_rate'],
            'step': log['step'],
        })
    df = pd.DataFrame(result)
    return df


# In[4]:


def all_spp_accuracy(loaded_reports) -> pd.DataFrame:
    result = []
    for key, report in loaded_reports.items():
        result.append({
            "checkpoint": key,
            "accuracy": report["accuracy"]
        })
    return pd.DataFrame(result)


# In[5]:


def macro_avg(loaded_reports) -> pd.DataFrame:
    result = []
    for key, report in loaded_reports.items():
        result.append({
            "checkpoint": key,
            "precision": report["macro avg"]["precision"],
            "recall": report["macro avg"]["recall"],
            "f1-score": report["macro avg"]["f1-score"],
            "support": report["macro avg"]["support"]
        })
    return pd.DataFrame(result)


# In[6]:


def main():
    root_path = "../models/15spp_zoom_level_validation_models/1_seed_model_20250127"
    classification_reports = load_classification_reports(root_path)
    train_state_df = train_state_to_df(root_path + "/trainer_state.json")

    # print(f"Loaded {len(classification_reports)} classification reports")
    # for file_name, report in classification_reports.items():
    #     print(f"File: {file_name}")
    #     print(json.dumps(report, indent=4))
    #     print("\n\n")
    #     break
    # print(json.dumps(classification_reports['500'], indent=4))

    # all_spp_accuracy_df = all_spp_accuracy(classification_reports)
    # print(all_spp_accuracy_df)

    print(train_state_df)

main()

