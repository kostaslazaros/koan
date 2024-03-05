from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from . import borda_functions as bdf
from .plotting import plot_borda_importance


def borda_importance(df, tagname, sampling_method, results_path):
    tag = df[[tagname]]
    df = df.drop(columns=[tagname])

    xgb = XGBClassifier(
        eval_metric="logloss",
        verbosity=0,
        importance_type="gain",
        use_label_encoder=False,
        objective="binary:logistic",
    )
    cbc = CatBoostClassifier(iterations=100, verbose=0)
    lgbm = LGBMClassifier(importance_type="split", objective="binary")

    xgb.fit(df, tag[tagname])
    cbc.fit(df, tag[tagname])
    lgbm.fit(df, tag[tagname])

    xgb_importance = bdf.feature_importance_df_creation(
        xgb, "XgBoost_importance", list(df.columns)
    )
    cbc_importance = bdf.feature_importance_df_creation(
        cbc, "CatBoost_importance", list(df.columns)
    )
    lgbm_importance = bdf.feature_importance_df_creation(
        lgbm, "LightGBM_importance", list(df.columns)
    )

    xgboost_imp_lst = list(xgb_importance["feature_name"])
    catboost_imp_lst = list(cbc_importance["feature_name"])
    lgbm_imp_lst = list(lgbm_importance["feature_name"])

    borda_results = bdf.borda_df([xgboost_imp_lst, catboost_imp_lst, lgbm_imp_lst])
    # borda_results, sampling_type, results_path
    plot_borda_importance(borda_results, sampling_method, results_path)
    return {
        "borda_importance": borda_results,
        "tag": tag,
        "df": df,
        "feature_list": list(borda_results["Feature"]),
    }
