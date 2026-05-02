import pandas as pd


def min_max_scale(df):
    def min_max_scale(group):
        min_val = group.min()
        max_val = group.max()
        if max_val == min_val:
            return pd.Series(100.0, index=group.index)
        return ((max_val - group) / (max_val - min_val)) * 100.0

    df["metric_error"] = df.groupby(["tid", "metric"])["metric_error"].transform(min_max_scale)
    return df


def z_score_scale(df):
    def z_score_scale(group):
        mean_val = group.mean()
        std_val = group.std()

        # If all methods perform identically, standard deviation is 0
        if std_val == 0 or pd.isna(std_val):
            return pd.Series(0.0, index=group.index)

        return (group - mean_val) / std_val

    # Apply the Z-score transformation group by group
    df["scaled_score"] = df.groupby("feature_selection_method")["metric_error"].transform(z_score_scale)
    return df


def median_max_scale(df):
    def robust_scale(group):
        min_val = group.max()
        median_val = group.median()

        # Edge case: If all methods perform identically, they all get 100
        if median_val == min_val:
            return pd.Series(100.0, index=group.index)

        return ((median_val - group) / (median_val - min_val)) * 100.0

    df["metric_error"] = df.groupby(["tid", "metric"])["metric_error"].transform(robust_scale)
    return df


def tabarena_normalization(df):
    def tabarena_scale(group):
        err_best = group.min()  # Lowest error is the best!
        err_median = group.median()

        if err_median == err_best:
            return pd.Series(1.0, index=group.index)  # Using 100.0 for percentages

        # Formula: (Median - Error) / (Median - Best)
        raw_score = (err_median - group) / (err_median - err_best)

        return raw_score.clip(lower=0)

    df["normalized_score"] = df.groupby(["tid", "metric"])["metric_error"].transform(tabarena_scale)
    return df
