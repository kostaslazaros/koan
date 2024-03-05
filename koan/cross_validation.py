"""Cross Validation Experiments"""

from sklearn.model_selection import KFold, cross_validate


def cross_val_exp(bordas, classifier_name, classifier, nsplit=10, tag_name="anyadr"):
    """
    Cross Validation Experiment
    bordas: {"borda_importance": borda_results, "tag": tag, "df": df}
    """
    kf = KFold(n_splits=nsplit, shuffle=True)
    accuracies = []
    f1s = []
    roc_aucs = []
    # Loop over the feature list and run cross-validation
    for i in range(len(bordas["feature_list"])):
        selected_features = bordas["feature_list"][: i + 1]
        selected_df = bordas["df"][selected_features]
        class_scores = cross_validate(
            classifier,
            selected_df,
            bordas["tag"][tag_name],
            scoring=("accuracy", "f1", "roc_auc"),
            cv=kf,
        )

        accuracies.append(class_scores["test_accuracy"])
        f1s.append(class_scores["test_f1"])
        roc_aucs.append(class_scores["test_roc_auc"])

    return {
        "classifier_name": classifier_name,
        "scores": {"Accuracy": accuracies, "F1-score": f1s, "ROC-AUC": roc_aucs},
    }
