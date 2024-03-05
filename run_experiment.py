import pathlib
from koan.workflows import workflow
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression as LRC

BASE_PATH = pathlib.Path("/home/kostas/prj/polypharmacy_project/datasets/anyadr_tag")
RESULTS_PATH = BASE_PATH / "results"
csv_file = BASE_PATH / "greek_pts_numerical_thyroid_anyadr_tag_v1.csv"
tagvals = {0.0: "No", 1.0: "Yes"}
INTEGER_COLUMNS = [
    "SEX",
    "AGE",
    "SMOKING",
    "ALC",
    "INDDRUG1",
    "INDICAT_GROUP",
    "DOSSCHED1",
    "DRUGACT1",
    "INDDRUG2",
    "DOSSCHED2",
    "Thyroid Malfunction",
    "primary",
    "Hospitalized",
    "anyadr",
]
CLASSIFIERS = {
    "LDA": LDA(),
    "KNN": KNN(n_neighbors=10),
    "LRC": LRC(),
}

for smethod in ["no", "undersampling", "smote"]:
    workflow(
        csvfile=csv_file,
        tagname="anyadr",
        sampling_method=smethod,
        tag_vals=tagvals,
        results_path=RESULTS_PATH,
        int_columns=INTEGER_COLUMNS,
        classifiers=CLASSIFIERS,
    )
