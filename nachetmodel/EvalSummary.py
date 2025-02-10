import os
import json
import pandas as pd


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

def all_spp_accuracy(loaded_reports) -> pd.DataFrame:
    result = []
    for key, report in loaded_reports.items():
        result.append({
            "checkpoint": key,
            "accuracy": report["accuracy"]
        })
    return pd.DataFrame(result)

def train_state_to_df(train_state) -> pd.DataFrame:
    df = pd.DataFrame(train_state)
    return df

def main():
    root_path = "models/15spp_zoom_level_validation_models/1_seed_model_20250127"
    classification_reports = load_classification_reports(root_path)
    print(json.dumps(classification_reports['500'], indent=4))

    train_state_df = train_state_to_df(root_path + "/trainer_state.json")


if __name__ == "__main__":
    main()
