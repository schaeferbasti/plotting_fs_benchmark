def average_per_budget_and_method(df):
    df = df.groupby(
        ["feature_selection_budget_index", "metric", "feature_selection_method"]
    )[["metric_error", "feature_selection_fit_time"]].mean().reset_index()
    return df

def average_per_model_and_method(df):
    df = df.groupby(
        ["model_name", "metric", "feature_selection_method"]
    )[["metric_error", "feature_selection_fit_time"]].mean().reset_index()
    return df

def average_per_dataset_and_method(df):
    df = df.groupby(
        ["tid", "metric", "feature_selection_method"]
    )[["metric_error", "feature_selection_fit_time"]].mean().reset_index()
    return df

def average_per_method(df):
    df = df.groupby(
        ["metric", "feature_selection_method"]
    )[["metric_error"]].mean().reset_index()
    return df
