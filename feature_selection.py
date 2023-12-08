import os
from typing import List

import xgboost
from joblib import Parallel, delayed
import contextlib
import joblib

import numpy as np
import pandas as pd
import lightgbm as lgb
from joblib import parallel_backend

import warnings

from tqdm import tqdm

warnings.filterwarnings("ignore")

from sklearn import (
    linear_model,
    pipeline,
    preprocessing,
    model_selection,
    feature_selection,
    ensemble,
    metrics,
    neural_network,
)

from helper import evaluate_performance

import multiprocessing

NUM_CORES = multiprocessing.cpu_count()


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def meanstd_str(arr, prec=3) -> str:
    return f"{np.nanmean(arr):.{prec}f}±{np.nanstd(arr):.{prec}f}"


def get_pipeline():
    return pipeline.Pipeline(
        [
            ("scale", preprocessing.StandardScaler()),
            (
                "predict",
                linear_model.LogisticRegression(C=1e-2, class_weight="balanced"),
            ),
        ]
    )


def calculate_cv_score(
    X,
    y,
    features,
    groups,
    scenarios,
    train_indices,
    test_indices,
    desc,
    bg=None,
    store_files=False,
):
    train_type = (
        "moderate"
        if np.all(np.unique(groups[train_indices]) < 300)
        else "mild"
        if np.all(np.unique(groups[train_indices]) > 300)
        else "mixed"
    )
    eval_type = (
        "moderate"
        if np.all(np.unique(groups[test_indices]) < 300)
        else "mild"
        if np.all(np.unique(groups[test_indices]) > 300)
        else "mixed"
    )
    print(
        f"### Running {desc.upper()}: train {train_type.upper()}, evaluate {eval_type.upper()}."
    )
    # print(f'Train subjects (n = {len(groups[train_indices].unique())}): {sorted(groups[train_indices].unique())}')
    # print(f'Test subjects (n = {len(groups[test_indices].unique())}): {sorted(groups[test_indices].unique())}')
    print(f"Features (n = {len(features)}): {features}")

    pipe = get_pipeline()
    aucs = []
    results = pd.DataFrame(
        {
            "id": groups[test_indices],
            "y_test": y[test_indices],
            "scenario": scenarios[test_indices],
        },
        index=y[test_indices].index,
    )
    coefs = pd.DataFrame(index=pd.Index(np.unique(groups[test_indices]), name="id"))

    # fit only once if no test subject in train sets
    fitted = False
    if (
        len(
            np.intersect1d(
                np.unique(groups[train_indices]), np.unique(groups[test_indices])
            )
        )
        == 0
    ):
        pipe.fit(X[train_indices][features], y[train_indices])
        fitted = True

    subjects = (
        np.unique(groups[test_indices])
        if fitted
        else tqdm(np.unique(groups[test_indices]))
    )

    for group in subjects:
        if not fitted:
            pipe.fit(
                X[train_indices & (groups != group)][features],
                y[train_indices & (groups != group)],
            )

        y_pred = pipe.predict_proba(X[test_indices & (groups == group)][features])[:, 1]
        y_true = y[test_indices & (groups == group)]

        results.loc[groups == group, "y_pred"] = y_pred

        aucs.append(
            metrics.roc_auc_score(y_true, y_pred)
        )  # if np.unique(y_true).size == 2 else np.nan)

        if hasattr(pipe["predict"], "coef_"):
            coefs.loc[group, features] = pipe["predict"].coef_

    print(f"AUC: {meanstd_str(aucs)}")
    evaluate_performance(
        results["y_test"],
        results["y_pred"],
        results["id"],
        results["scenario"],
        print_df=True,
        print_csv=True,
        threshold=-1,
        name=f"{desc}_{eval_type}",
    )
    print()

    if store_files:
        filename = f"{desc}_train_{train_type}_eval_{eval_type}"
        results.to_pickle(f"output/{filename}.pkl")
        coefs.to_pickle(f"output/{filename}_coefs.pkl")
