"""
Utility functions
"""

import os
import datetime
import shutil


def create_save_path(subdir, filename, root_path):
    now = datetime.datetime.now()
    formatted_now = now.strftime(f"%Y-%m-%d_{subdir}")
    dir_path = os.path.join(root_path, formatted_now)
    os.makedirs(dir_path, exist_ok=True)
    return os.path.join(dir_path, filename)


def move_file_to_subdir(file_name, subdir, root_path):
    dest_path = create_save_path(subdir, file_name, root_path)
    shutil.move(file_name, dest_path)


def dataframe_columns2integer(df, columns: list):
    df[columns] = df[columns].astype(int)
    return df
