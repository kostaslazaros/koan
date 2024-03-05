import pandas as pd
from .utilities import dataframe_columns2integer
from .sampling import sampling


def preprocess(
    csvfile: str,
    int_columns: list,
    tagname: str,
    sampling_method: str,
) -> pd.DataFrame:
    """
    sampling_method: ["no", "undersampling", "smote"]
    """
    df = pd.read_csv(csvfile, index_col=0)
    df = dataframe_columns2integer(df, int_columns)
    df = sampling(df, tag_name=tagname, typos=sampling_method)
    return df
