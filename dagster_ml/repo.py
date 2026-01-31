from dagster import Definitions

from dagster_ml.assets.pipeline import (
    raw_data,
    eda_summary,
    train_test,
    decision_tree,
    random_forest,
    logistic_regression,
    knn,
)

defs = Definitions(
    assets=[
        raw_data,
        eda_summary,
        train_test,
        decision_tree,
        random_forest,
        logistic_regression,
        knn,
    ]
)
