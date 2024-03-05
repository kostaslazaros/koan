"""
functions to do sampling
"""

import pandas as pd
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE


def _undersampling(df, tag_name="anyadr"):
    # Separate the majority and minority classes
    class_distro = dict(df[tag_name].value_counts())
    lst = list(class_distro.values())
    ks = list(class_distro.keys())
    df_minority = df[df[tag_name] == ks[lst.index(min(lst))]]
    df_majority = df[df[tag_name] == ks[lst.index(max(lst))]]
    # Find the number of observations in the minority class
    n_minority = len(df_minority)
    # Downsample the majority class to match the minority class size
    df_majority_downsampled = resample(
        df_majority, replace=False, n_samples=n_minority  # sample without replacement
    )  # to match minority class
    df_balanced = pd.concat([df_majority_downsampled, df_minority])
    return df_balanced


def _smote_oversampling(df, tag_name="anyadr"):
    # Separate features and target variable
    X = df.drop(tag_name, axis=1)  # Features
    y = df[tag_name]  # Target variable
    # Initialize SMOTE
    smote = SMOTE(random_state=42)
    # Apply SMOTE to your data
    df_balanced, tag_resampled = smote.fit_resample(X, y)
    df_balanced[tag_name] = tag_resampled
    return df_balanced


def sampling(df, *, tag_name, typos):
    """
    no            : Do nothing
    undersampling : undersample
    smote         : oversampling
    """
    if typos == "no":
        return df

    if typos == "undersampling":
        return _undersampling(df, tag_name)

    if typos == "smote":
        return _smote_oversampling(df, tag_name)

    raise ValueError(f"Not existing typos: {typos}")
