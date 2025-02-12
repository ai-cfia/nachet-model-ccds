#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import json
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


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


# In[ ]:


def train_state_to_df(train_state) -> pd.DataFrame:
    result = []

    with open(train_state, "r") as f:
        state_json = json.load(f)
        for log in state_json["log_history"]:
            try:
                result.append(
                    {
                        "epoch": log["epoch"],
                        "grad_norm": log["grad_norm"],
                        "loss": log["loss"],
                        "learning_rate": log["learning_rate"],
                        "step": log["step"],
                    }
                )
            except KeyError:
                pass
    df = pd.DataFrame(result)
    return df


# In[ ]:


def plot_loss(df: pd.DataFrame, save: bool = False, save_path: str = "./") -> None:
    fig, ax = plt.subplots()
    ax.plot(df["epoch"], df["loss"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.set_ylim(top=0.05, bottom=-0.0001)
    
    # ax.ylim(top=df['loss'].max())
    fig.savefig(f"{save_path}loss_plot.png") if save else fig.show()
    plt.close(fig)


# In[ ]:


def checkpoint_to_epoch(state: pd.DataFrame, checkpoint: int) -> int:
    if checkpoint % 500 != 0:
        return -1
    return state[state["step"] == checkpoint]["epoch"].values[0]


# In[ ]:


def all_spp_accuracy(loaded_reports) -> pd.DataFrame:
    result = []
    for key, report in loaded_reports.items():
        result.append({"checkpoint": key, "accuracy": report["accuracy"]})
    return pd.DataFrame(result)


# In[ ]:


def plot_accuracy(df: pd.DataFrame, state: pd.DataFrame, title: str, save: bool = False, save_path: str = "./") -> None:

    # use epoch as x-axis
    df["epoch"] = df["checkpoint"].apply(lambda x: checkpoint_to_epoch(state, int(x)))
    # sort by epoch
    df = df.sort_values(by="epoch")

    fig, ax = plt.subplots()
    ax.plot(df["epoch"], df["accuracy"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.set_ylim(top=1.0, bottom=0.0)
    fig.savefig(f"{save_path}{title}_accuracy_plot.png") if save else fig.show()
    plt.close(fig)


# In[ ]:


def plot_metric(df: pd.DataFrame, state: pd.DataFrame, title: str, metric: str, save: bool = False, save_path: str = "./") -> None:
    # use epoch as x-axis
    df["epoch"] = df["checkpoint"].apply(lambda x: checkpoint_to_epoch(state, int(x)))
    # sort by epoch
    df = df.sort_values(by="epoch")

    fig, ax = plt.subplots()
    ax.plot(df["epoch"], df[metric])
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric)
    ax.set_title(title)
    fig.savefig(f"{save_path}{title}_{metric}_plot.png") if save else fig.show()
    plt.close(fig)


# In[ ]:


def macro_avg(loaded_reports) -> pd.DataFrame:
    result = []
    for key, report in loaded_reports.items():
        result.append(
            {
                "checkpoint": key,
                "precision": report["macro avg"]["precision"],
                "recall": report["macro avg"]["recall"],
                "f1-score": report["macro avg"]["f1-score"],
                "support": report["macro avg"]["support"],
            }
        )
    return pd.DataFrame(result)


# In[ ]:


# def plot_class_accuracy(loaded_reports, state, class_name):
#     result = []
#     for key, report in loaded_reports.items():
#         result.append(
#             {
#                 "checkpoint": key,
#                 "precision": report[class_name]["precision"],
#                 "recall": report[class_name]["recall"],
#                 "f1-score": report[class_name]["f1-score"],
#                 "support": report[class_name]["support"],
#                 "accuracy": report[class_name]["accuracy"],
#             }
#         )
#     df = pd.DataFrame(result)
#     plot_accuracy(df, state, class_name)


# In[ ]:


def plot_class_metric(
    loaded_reports: pd.DataFrame,
    state: pd.DataFrame,
    title: str,
    metric: str,
    class_name: str,
    save: bool = False,
    save_path: str = "./",
):
    result = []
    for key, report in loaded_reports.items():
        result.append(
            {
                "checkpoint": key,
                "precision": report[class_name]["precision"],
                "recall": report[class_name]["recall"],
                "f1-score": report[class_name]["f1-score"],
                "support": report[class_name]["support"],
                "accuracy": report[class_name]["accuracy"],
            }
        )
    df = pd.DataFrame(result)
    plot_metric(df, state, title, metric, save, save_path)


# In[ ]:


def move_checkpoint_plots(chk_root: str, save_path: str) -> None:
    for subdir, dirs, files in os.walk(chk_root):
        for file_name in files:
            if file_name.endswith(".png") and "checkpoint" in subdir:
                full_path = os.path.join(subdir, file_name)
                os.rename(full_path, f"{save_path}{file_name}")


# In[ ]:


def get_class_names(loaded_reports) -> list:
    result = []
    for key in loaded_reports["500"]:
        if key not in ["accuracy", "macro avg", "weighted avg"]:
            result.append(key)
    return result


# In[ ]:


def process_model(root_path: str, save: bool = False) -> None:
    save_path = f"{root_path}/plots/"
    if not os.path.exists(root_path + "/plots"):
        os.makedirs(root_path + "/plots")

    loaded_reports = load_classification_reports(root_path)
    state = train_state_to_df(f"{root_path}/trainer_state.json")
    
    class_names = get_class_names(loaded_reports)
    all_spp = all_spp_accuracy(loaded_reports)

    plot_loss(state, save, save_path)
    plot_accuracy(all_spp, state, "All Species", save, save_path)
    
    for class_name in class_names:
        plot_class_metric(loaded_reports, state, class_name, "accuracy", class_name, save, save_path)
        plot_class_metric(loaded_reports, state, class_name, "precision", class_name, save, save_path)
        plot_class_metric(loaded_reports, state, class_name, "recall", class_name, save, save_path)
        plot_class_metric(loaded_reports, state, class_name, "f1-score", class_name, save, save_path)
    
    move_checkpoint_plots(root_path, save_path)


# In[ ]:


def main():
    # process_model("../models/15spp_zoom_level_validation_models/1_seed_model_20250127", save=True)
    # print("1/7 Done")

    # process_model("../models/15spp_zoom_level_validation_models/1_seed_model_20250128", save=True)
    # print("2/7 Done")
    
    # process_model("../models/15spp_zoom_level_validation_models/1_seed_model_20250129", save=True)
    # print("3/7 Done")
    
    process_model("../models/15spp_zoom_level_validation_models/2_seed_model_120250130", save=True)
    print("4/7 Done")
    
    process_model("../models/15spp_zoom_level_validation_models/6_seed_model_120250130", save=True)
    print("5/7 Done")
    
    process_model("../models/27spp_model/model_120250130", save=True)
    print("6/7 Done")
    
    process_model("../models/27spp_model/model_220250130", save=True)
    print("7/7 Done")

main()

