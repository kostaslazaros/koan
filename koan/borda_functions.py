import pandas as pd


# Function to assign a rank to each element in a list with the first element being the most important.
def rank_list(lst: list) -> dict:
    # Returns a dictionary comprehension where each list element is a key and its rank is a value.
    # The rank is calculated as the length of the list minus one minus the element's index.
    return {elm: len(lst) - 1 - i for i, elm in enumerate(lst)}


# Function to perform Borda count aggregation for a list of ranked lists.
def borda_aggregation(loflists: list[list]) -> dict:
    # Convert each individual list into a dictionary of ranks.
    list_ranks = [rank_list(l) for l in loflists]
    # Create a set of all unique elements across all the lists.
    feature_set = {i for i in [el for nl in loflists for el in nl]}
    # Return a dictionary where each element's score is the sum of its ranks across all the lists.
    return {e: sum([lr.get(e, 0) for lr in list_ranks]) for e in feature_set}


# Function to create a sorted DataFrame from a dictionary of results.
def create_sorted_df(result: dict):
    # Create a DataFrame from the dictionary.
    df_sorted = pd.DataFrame(list(result.items()), columns=["Feature", "Borda Rank"])
    # Sort the DataFrame based on the 'Borda Rank' column in descending order.
    df_sorted.sort_values(by="Borda Rank", ascending=False, inplace=True)
    # Reset the DataFrame's index and drop the old index.
    df_sorted.reset_index(drop=True, inplace=True)
    # Return the sorted DataFrame.
    return df_sorted


# Function to create a DataFrame from a list of lists using Borda count aggregation.
def borda_df(loflists: list[list]) -> pd.DataFrame:
    # Aggregate the lists into a Borda count dictionary.
    borda_dict = borda_aggregation(loflists)
    # Create and return a sorted DataFrame from the Borda count dictionary.
    return create_sorted_df(borda_dict)


# Function to create a DataFrame showing the importance of features from a model.
def feature_importance_df_creation(model, model_label, feature_names):
    # Create a DataFrame from the model's feature importances.
    df_importance = pd.DataFrame(model.feature_importances_)
    # Rename the column to the provided model label.
    df_importance = df_importance.rename(columns={0: model_label})
    # Add the feature names as a column in the DataFrame.
    df_importance["feature_name"] = feature_names
    # Sort the DataFrame based on the importance scores in descending order.
    df_importance = df_importance.sort_values(by=model_label, ascending=False)
    # Return the sorted DataFrame.
    return df_importance
