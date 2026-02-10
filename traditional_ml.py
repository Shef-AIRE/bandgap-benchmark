import json
import os
from typing import Any, Dict, Optional

import joblib
import numpy as np
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    make_scorer,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVR

from realmat_bag.loaddata.dataloader import extract_features
from realmat_bag.pipeline.trainer import mean_relative_error


def _build_estimator(model_name: str, seed: int) -> Any:
    if model_name == "random_forest":
        return RandomForestRegressor(n_estimators=100, random_state=seed)
    if model_name == "linear_regression":
        return LinearRegression()
    if model_name == "svm":
        # kernel will be set via hyperparams if provided
        return SVR()
    raise ValueError(f"Unsupported traditional model: {model_name}")


def _parse_hyperparams(cfg) -> Dict[str, Any]:
    hyper_cfg = getattr(cfg.MODEL, "HYPERPARAMS", None)
    if hyper_cfg is None:
        return {}
    try:
        keys = list(hyper_cfg.keys())
    except AttributeError:
        return {}
    if not keys:
        return {}
    try:
        hyper_dict = yaml.safe_load(hyper_cfg.dump())
    except AttributeError:
        hyper_dict = dict(hyper_cfg)
    return hyper_dict


def run_traditional_model(cfg, train_dataset, val_dataset, fold_label: str, test_dataset=None):
    model_name = cfg.MODEL.NAME
    seed = cfg.SOLVER.SEED

    X_train, y_train = extract_features(train_dataset)
    X_val, y_val = extract_features(val_dataset)

    base_estimator = _build_estimator(model_name, seed)

    hyper_cfg = _parse_hyperparams(cfg)
    search_type = hyper_cfg.pop("SEARCH", hyper_cfg.pop("search", "grid")).lower() if hyper_cfg else "grid"
    n_iter = hyper_cfg.pop("N_ITER", hyper_cfg.pop("n_iter", 10)) if hyper_cfg else 10
    inner_cv = hyper_cfg.pop("CV", hyper_cfg.pop("cv", 3)) if hyper_cfg else 3
    param_grid = hyper_cfg if hyper_cfg else None

    best_params = None
    if param_grid:
        scorer = make_scorer(mean_absolute_error, greater_is_better=False)
        if search_type == "random":
            search = RandomizedSearchCV(
                base_estimator,
                param_distributions=param_grid,
                n_iter=n_iter,
                scoring=scorer,
                cv=inner_cv,
                random_state=seed,
                n_jobs=-1,
            )
        else:
            search = GridSearchCV(
                base_estimator,
                param_grid=param_grid,
                scoring=scorer,
                cv=inner_cv,
                n_jobs=-1,
            )
        search.fit(X_train, y_train)
        model = search.best_estimator_
        best_params = search.best_params_
    else:
        model = base_estimator
        model.fit(X_train, y_train)

    safe_label = fold_label.replace(os.sep, "_")
    os.makedirs(cfg.LOGGING.LOG_DIR, exist_ok=True)
    feature_importances_file = os.path.join(
        cfg.LOGGING.LOG_DIR, f"feature_importances_{safe_label}.txt"
    )

    relative_error_scorer = make_scorer(mean_relative_error, greater_is_better=False)
    importance_result = permutation_importance(
        estimator=model,
        X=X_val,
        y=y_val,
        scoring=relative_error_scorer,
        n_repeats=10,
        random_state=seed,
    )
    importances_mean = importance_result.importances_mean
    with open(feature_importances_file, "w") as f:
        f.write("Feature Importances:\n")
        for idx, imp in enumerate(importances_mean):
            f.write(f"Feature {idx:2d}: {imp}\n")

    model_path = os.path.join(cfg.LOGGING.LOG_DIR, f"model_{safe_label}.joblib")
    joblib.dump(model, model_path)

    best_params_file = None
    if best_params:
        best_params_file = os.path.join(
            cfg.LOGGING.LOG_DIR, f"best_params_{safe_label}.json"
        )
        with open(best_params_file, "w") as f:
            json.dump(best_params, f, indent=2)

    val_predictions = model.predict(X_val)
    val_mae = mean_absolute_error(y_val, val_predictions)
    val_mse = mean_squared_error(y_val, val_predictions)
    val_mre = np.mean(np.abs((y_val - val_predictions) / y_val))
    val_r2 = r2_score(y_val, val_predictions)

    test_metrics: Optional[Dict[str, float]] = None
    if test_dataset is not None:
        X_test, y_test = extract_features(test_dataset)
        test_predictions = model.predict(X_test)
        test_mae = mean_absolute_error(y_test, test_predictions)
        test_mse = mean_squared_error(y_test, test_predictions)
        test_mre = np.mean(np.abs((y_test - test_predictions) / y_test))
        test_r2 = r2_score(y_test, test_predictions)
        test_metrics = {
            "mae": test_mae,
            "mse": test_mse,
            "mre": test_mre,
            "r2": test_r2,
        }

    val_metrics = {"mae": val_mae, "mse": val_mse, "mre": val_mre, "r2": val_r2}

    # print results and best params
    print(f"Results for fold {fold_label}:")
    print(f"  Validation MAE: {val_mae:.6f}")
    print(f"  Validation MSE: {val_mse:.6f}")
    print(f"  Validation MRE: {val_mre:.6f}")
    print(f"  Validation R2:  {val_r2:.6f}")
    if test_metrics:
        print(f"  Test MAE: {test_metrics['mae']:.6f}")
        print(f"  Test MSE: {test_metrics['mse']:.6f}")
        print(f"  Test MRE: {test_metrics['mre']:.6f}")
        print(f"  Test R2:  {test_metrics['r2']:.6f}")
    if best_params:
        print(f"  Best Hyperparameters: {best_params}")

    return {
        "model": model,
        "best_params": best_params,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "feature_importances_path": feature_importances_file,
        "best_params_path": best_params_file,
        "model_path": model_path,
    }
