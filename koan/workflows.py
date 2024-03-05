import warnings

warnings.filterwarnings("ignore")
from . import plotting as plo
from .preprocess import preprocess
from .pycaret import exec_pycaret
from .borda_importance import borda_importance
from .cross_validation import cross_val_exp


def workflow(
    *,
    csvfile,
    tagname,
    sampling_method,
    tag_vals,
    results_path,
    int_columns,
    classifiers
):
    """
    csvfile: csv file containing dataset
    tagname: anyaddr or any other tag_name
    sampling_method: can take one of the values ["no", "undersampling", "smote"]
    tag_vals: dictionary of type {0.0: "No", 1.0: "Yes"}
    results_path: root path to save results (data and images)
    """

    df = preprocess(
        csvfile,
        int_columns,
        tagname=tagname,
        sampling_method=sampling_method,
    )
    plo.plot_dimensionality_reduction(df, tag_vals, sampling_method, results_path)
    plo.plot_correlation_heatmap(df, sampling_method, results_path)
    exec_pycaret(df, tagname, sampling_method, results_path)
    # {"borda_importance": borda_results, "tag": tag, "df": df}
    borda_results = borda_importance(df, tagname, sampling_method, results_path)
    for classifier_name, classifier in classifiers.items():
        cross_res = cross_val_exp(
            borda_results, classifier_name, classifier, tag_name=tagname
        )
        plo.plot_boxplot_metrics(cross_res, sampling_method, results_path)
