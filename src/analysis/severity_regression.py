import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.analysis.utils import conversation_train_test_split, load_transcript_rows, save_metrics

TARGET = 'eval_full_violation'
BASE_FEATURES = ['judge_refusal', 'judge_partial_compliance', 'judge_full_violation', 'timestep_in_conversation']


def build_features(df: pd.DataFrame):
    X = df[BASE_FEATURES].copy()
    X = pd.concat([X, pd.get_dummies(df['technique'], prefix='technique')], axis=1)
    X = pd.concat([X, pd.get_dummies(df['target_model'], prefix='target')], axis=1)
    return X


def evaluate(name: str, model, X_train, y_train, X_test, y_test) -> dict:
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = {'rmse': float(np.sqrt(mean_squared_error(y_test, preds))), 'mae': float(mean_absolute_error(y_test, preds)), 'r2': float(r2_score(y_test, preds))}
    save_metrics(f'regression_{name}', metrics)
    return metrics


def main():
    df = load_transcript_rows()
    train_df, test_df = conversation_train_test_split(df)
    X_train_raw, X_test_raw = (build_features(train_df), build_features(test_df))
    X_train, X_test = X_train_raw.align(X_test_raw, join='outer', axis=1, fill_value=0)
    y_train, y_test = (train_df[TARGET].values, test_df[TARGET].values)
    print(f'\n=== regression: predicting {TARGET} from live-judge signal + turn metadata ===')
    evaluate('linear_regression', LinearRegression(), X_train, y_train, X_test, y_test)
    evaluate('random_forest', RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42), X_train, y_train, X_test, y_test)


if __name__ == '__main__':
    main()
