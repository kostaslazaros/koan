from pycaret import classification as clf
from .utilities import move_file_to_subdir


def exec_pycaret(df, tag_name, sampling_type, result_path):

    s = clf.setup(df, target=tag_name, session_id=123)
    best = clf.compare_models()

    clf.plot_model(best, plot="confusion_matrix", save=True)
    move_file_to_subdir("Confusion Matrix.png", sampling_type, result_path)

    clf.plot_model(best, plot="error", save=True)
    move_file_to_subdir("Prediction Error.png", sampling_type, result_path)

    clf.plot_model(best, plot="class_report", save=True)
    move_file_to_subdir("Class Report.png", sampling_type, result_path)
