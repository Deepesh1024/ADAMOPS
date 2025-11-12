import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

from adamops import (
    DataLoader, MissingValueHandler, DataSplitter,
    FeatureScaler, FeatureEncoder, ModelTrainer,
    ModelEvaluator, LocalDeployer, ModelConfig
)

import pandas as pd

TARGET_COLUMN = "Survived"
DATA_PATH = "data/titanic.csv"


def run_end_to_end_pipeline():
    print("\n" + "="*60)
    print("TITANIC SURVIVAL PREDICTION - PIPELINE")
    print("="*60 + "\n")

    print("[1/8] Loading data...")
    df = DataLoader({'source_type': 'csv', 'source_path': DATA_PATH}).execute()

    print("[2/8] Handling missing values...")
    df_clean = MissingValueHandler({'strategy': 'mean', 'drop_threshold': 0.6}).execute(df)

    print("[3/8] Splitting data...")
    splits = DataSplitter({
        'test_size': 0.2,
        'val_size': 0.1,
        'stratify': True,
        'random_state': 42
    }).execute(df_clean, target_col=TARGET_COLUMN)

    print("[4/8] Scaling features...")
    scaled = FeatureScaler({'method': 'standard'}).execute(
        X_train=splits['X_train'],
        X_val=splits.get('X_val'),
        X_test=splits['X_test']
    )

    print("[5/8] Encoding features...")
    encoded = FeatureEncoder({'method': 'label'}).execute(
        X_train=scaled['X_train'],
        X_val=scaled.get('X_val'),
        X_test=scaled['X_test']
    )

    print("[6/8] Training model...")
    model = ModelTrainer(ModelConfig(
        model_type='random_forest',
        hyperparameters={'n_estimators': 200, 'max_depth': 8},
        random_state=42
    )).execute(
        X_train=encoded['X_train'],
        y_train=splits['y_train'],
        X_val=encoded.get('X_val'),
        y_val=splits.get('y_val')
    )

    print("[7/8] Evaluating model...")
    metrics = ModelEvaluator({}).execute(
        model=model,
        X_test=encoded['X_test'],
        y_test=splits['y_test'],
        task_type='classification'
    )

    print("[8/8] Deploying model...")
    model_path = LocalDeployer({}).execute(model, model_name='titanic_survival')

    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print(f"Model saved: {model_path}")
    print("FINAL METRICS:")
    for k, v in metrics.items():
        print(f"  {k:12}: {v:.4f}")
    print("="*60)

    return model, metrics


if __name__ == "__main__":
    run_end_to_end_pipeline()