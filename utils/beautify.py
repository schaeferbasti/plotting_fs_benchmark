import ast

import pandas as pd


def beautify_names(df):
    df["feature_selection_method"] = df["feature_selection_method"].str.replace(
        "FeatureSelector", "", regex=False
    )

    # Rename specific methods to their acronyms/new names
    df["feature_selection_method"] = df["feature_selection_method"].replace({
        "Accuracy": "LOCO",
        "SequentialBackwardElimination": "RFE",
        "SequentialForwardSelection": "SFS",
        "ANOVA": "F-Test",
        "ReliefF": "(R)ReliefF"
    })
    return df


def remove_jmi(df):
    df = df[df["feature_selection_method"] != "JMI"]
    df = df[df["feature_selection_method"] != "AccuracyLinear"]
    return df


def remove_jmi_random(df):
    return df[~df["feature_selection_method"].isin(["JMI", "Random"])]


def add_model_name(df):
    def extract_model_cls(model_details):
        if pd.isna(model_details):
            return "Unknown"
        details_dict = ast.literal_eval(str(model_details))
        return details_dict.get('model_cls', "Unknown")

    df["model_cls"] = df["model_details"].apply(extract_model_cls)
    return df
