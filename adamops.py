from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Union
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
import sweetviz as sv
import json
import os
from pathlib import Path
import boto3
import docker
from datetime import datetime


class BaseOperation(ABC):
    def __init__(self, config: Dict[str, Any], name: str = None):
        self.config = config
        self.name = name or self.__class__.__name__
        self.metadata = {}
        self.timestamp = datetime.now().isoformat()
        
    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        pass
    
    def validate_config(self) -> bool:
        required_keys = getattr(self, 'REQUIRED_CONFIG_KEYS', [])
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")
        return True
    
    def log_to_mlflow(self, additional_params: Dict = None, additional_metrics: Dict = None):
        try:
            mlflow.log_params({f"{self.name}_{k}": v for k, v in self.config.items()})
            if additional_params:
                mlflow.log_params(additional_params)
            if self.metadata:
                mlflow.log_metrics({f"{self.name}_{k}": v for k, v in self.metadata.items() if isinstance(v, (int, float))})
            if additional_metrics:
                mlflow.log_metrics(additional_metrics)
        except Exception as e:
            print(f"MLflow logging warning: {e}")
    
    def save_artifact(self, data: Any, filename: str):
        Path("artifacts").mkdir(exist_ok=True)
        filepath = f"artifacts/{filename}"
        
        if isinstance(data, pd.DataFrame):
            data.to_csv(filepath, index=False)
        elif isinstance(data, dict):
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        
        try:
            mlflow.log_artifact(filepath)
        except:
            pass


class DataLoader(BaseOperation):
    REQUIRED_CONFIG_KEYS = ['source_type']
    
    def execute(self, source_path: str = None) -> pd.DataFrame:
        source_path = source_path or self.config.get('source_path')
        source_type = self.config['source_type']
        
        if source_type == 'csv':
            df = pd.read_csv(source_path)
        elif source_type == 'excel':
            df = pd.read_excel(source_path)
        elif source_type == 'parquet':
            df = pd.read_parquet(source_path)
        elif source_type == 'json':
            df = pd.read_json(source_path)
        else:
            raise ValueError(f"Unsupported source type: {source_type}")
        
        self.metadata = {
            'n_rows': len(df),
            'n_columns': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
        }
        
        print(f"Loaded {len(df)} rows and {len(df.columns)} columns from {source_type}")
        return df


class DataValidator(BaseOperation):
    def execute(self, df: pd.DataFrame, expected_schema: Dict = None) -> Dict[str, Any]:
        validation_report = {
            'is_valid': True,
            'issues': [],
            'statistics': {}
        }
        
        missing_pct = (df.isnull().sum() / len(df) * 100).to_dict()
        validation_report['statistics']['missing_percentage'] = missing_pct
        
        if any(pct > self.config.get('max_missing_pct', 50) for pct in missing_pct.values()):
            validation_report['is_valid'] = False
            validation_report['issues'].append("Excessive missing values detected")
        
        n_duplicates = df.duplicated().sum()
        validation_report['statistics']['n_duplicates'] = int(n_duplicates)
        
        if n_duplicates > 0:
            validation_report['issues'].append(f"Found {n_duplicates} duplicate rows")
        
        if expected_schema:
            actual_cols = set(df.columns)
            expected_cols = set(expected_schema.keys())
            
            if actual_cols != expected_cols:
                validation_report['is_valid'] = False
                validation_report['issues'].append(f"Schema mismatch. Missing: {expected_cols - actual_cols}, Extra: {actual_cols - expected_cols}")
        
        self.metadata = validation_report['statistics']
        self.save_artifact(validation_report, f"validation_report_{self.timestamp}.json")
        
        print(f"Validation complete. Valid: {validation_report['is_valid']}")
        return validation_report


class DataSplitter(BaseOperation):
    def execute(self, df: pd.DataFrame, target_col: str) -> Dict[str, pd.DataFrame]:
        test_size = self.config.get('test_size', 0.2)
        val_size = self.config.get('val_size', 0.1)
        random_state = self.config.get('random_state', 42)
        stratify = self.config.get('stratify', False)
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        stratify_arg = y if stratify else None
        
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify_arg
        )
        
        if val_size > 0:
            val_ratio = val_size / (1 - test_size)  # ← FIXED: was 'Motel_size'
            stratify_arg_temp = y_temp if stratify else None
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=stratify_arg_temp
            )
        else:
            X_train, y_train = X_temp, y_temp
            X_val, y_val = None, None
        
        self.metadata = {
            'train_size': len(X_train),
            'val_size': len(X_val) if X_val is not None else 0,
            'test_size': len(X_test)
        }
        
        print(f"Data split: Train={len(X_train)}, Val={len(X_val) if X_val is not None else 0}, Test={len(X_test)}")
        
        result = {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test
        }
        
        if X_val is not None:
            result['X_val'] = X_val
            result['y_val'] = y_val
        
        return result


class AutoEDA(BaseOperation):
    def execute(self, df: pd.DataFrame) -> str:
        report_name = self.config.get('report_name', f"eda_report_{self.timestamp}")
        report_path = f"artifacts/{report_name}.html"
        
        Path("artifacts").mkdir(exist_ok=True)
        
        report = sv.analyze(df)
        report.show_html(report_path, open_browser=False)
        
        self.metadata = {
            'n_features': len(df.columns),
            'n_rows': len(df),
            'missing_cells': int(df.isnull().sum().sum()),
            'duplicate_rows': int(df.duplicated().sum())
        }
        
        try:
            mlflow.log_artifact(report_path)
        except:
            pass
        
        print(f"EDA report generated: {report_path}")
        return report_path


class FeatureAnalyzer(BaseOperation):
    def execute(self, df: pd.DataFrame, target_col: str = None) -> Dict[str, Any]:
        analysis = {
            'numeric_features': [],
            'categorical_features': [],
            'correlations': {},
            'statistics': {}
        }
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        analysis['numeric_features'] = numeric_cols
        analysis['categorical_features'] = categorical_cols
        
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            analysis['correlations'] = corr_matrix.to_dict()
            
            if target_col and target_col in numeric_cols:
                target_corr = corr_matrix[target_col].drop(target_col).to_dict()
                analysis['target_correlations'] = target_corr
        
        analysis['statistics'] = df.describe().to_dict()
        
        self.metadata = {
            'n_numeric': len(numeric_cols),
            'n_categorical': len(categorical_cols)
        }
        
        self.save_artifact(analysis, f"feature_analysis_{self.timestamp}.json")
        
        print(f"Feature analysis complete: {len(numeric_cols)} numeric, {len(categorical_cols)} categorical")
        return analysis


class MissingValueHandler(BaseOperation):
    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        strategy = self.config.get('strategy', 'mean')
        threshold = self.config.get('drop_threshold', 0.5)
        
        df_processed = df.copy()
        
        missing_pct = df_processed.isnull().sum() / len(df_processed)
        cols_to_drop = missing_pct[missing_pct > threshold].index.tolist()
        df_processed = df_processed.drop(columns=cols_to_drop)
        
        if strategy == 'mean':
            numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
            df_processed[numeric_cols] = df_processed[numeric_cols].fillna(df_processed[numeric_cols].mean())
        elif strategy == 'median':
            numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
            df_processed[numeric_cols] = df_processed[numeric_cols].fillna(df_processed[numeric_cols].median())
        elif strategy == 'mode':
            for col in df_processed.columns:
                df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])
        elif strategy == 'drop':
            df_processed = df_processed.dropna()
        
        self.metadata = {
            'columns_dropped': len(cols_to_drop),
            'rows_before': len(df),
            'rows_after': len(df_processed)
        }
        
        print(f"Missing values handled: {len(cols_to_drop)} columns dropped, {len(df) - len(df_processed)} rows removed")
        return df_processed


class FeatureScaler(BaseOperation):
    def __init__(self, config: Dict[str, Any], name: str = None):
        super().__init__(config, name)
        self.scaler = None
    
    def execute(self, X_train: pd.DataFrame, X_val: pd.DataFrame = None, X_test: pd.DataFrame = None) -> Dict[str, pd.DataFrame]:
        method = self.config.get('method', 'standard')
        
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        
        X_train_scaled = X_train.copy()
        X_train_scaled[numeric_cols] = self.scaler.fit_transform(X_train[numeric_cols])
        
        result = {'X_train': X_train_scaled}
        
        if X_val is not None:
            X_val_scaled = X_val.copy()
            X_val_scaled[numeric_cols] = self.scaler.transform(X_val[numeric_cols])
            result['X_val'] = X_val_scaled
        
        if X_test is not None:
            X_test_scaled = X_test.copy()
            X_test_scaled[numeric_cols] = self.scaler.transform(X_test[numeric_cols])
            result['X_test'] = X_test_scaled
        
        self.metadata = {
            'n_features_scaled': len(numeric_cols),
            'scaling_method': method
        }
        
        print(f"Features scaled using {method}: {len(numeric_cols)} features")
        return result


class FeatureEncoder(BaseOperation):
    def __init__(self, config: Dict[str, Any], name: str = None):
        super().__init__(config, name)
        self.encoders = {}
    
    def execute(self, X_train: pd.DataFrame, X_val: pd.DataFrame = None, X_test: pd.DataFrame = None) -> Dict[str, pd.DataFrame]:
        method = self.config.get('method', 'label')
        
        categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
        
        X_train_encoded = X_train.copy()
        
        for col in categorical_cols:
            if method == 'label':
                encoder = LabelEncoder()
                X_train_encoded[col] = encoder.fit_transform(X_train[col].astype(str))
                self.encoders[col] = encoder
            elif method == 'onehot':
                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                encoded_features = encoder.fit_transform(X_train[[col]])
                feature_names = [f"{col}_{cat}" for cat in encoder.categories_[0]]
                encoded_df = pd.DataFrame(encoded_features, columns=feature_names, index=X_train.index)
                X_train_encoded = X_train_encoded.drop(columns=[col])
                X_train_encoded = pd.concat([X_train_encoded, encoded_df], axis=1)
                self.encoders[col] = encoder
        
        result = {'X_train': X_train_encoded}
        
        if X_val is not None:
            X_val_encoded = self._transform(X_val, method, categorical_cols)
            result['X_val'] = X_val_encoded
        
        if X_test is not None:
            X_test_encoded = self._transform(X_test, method, categorical_cols)
            result['X_test'] = X_test_encoded
        
        self.metadata = {
            'n_features_encoded': len(categorical_cols),
            'encoding_method': method
        }
        
        print(f"Features encoded using {method}: {len(categorical_cols)} features")
        return result
    
    def _transform(self, X: pd.DataFrame, method: str, categorical_cols: List[str]) -> pd.DataFrame:
        X_encoded = X.copy()
        
        for col in categorical_cols:
            if col not in self.encoders:
                continue
            
            encoder = self.encoders[col]
            
            if method == 'label':
                X_encoded[col] = encoder.transform(X[col].astype(str))
            elif method == 'onehot':
                encoded_features = encoder.transform(X[[col]])
                feature_names = [f"{col}_{cat}" for cat in encoder.categories_[0]]
                encoded_df = pd.DataFrame(encoded_features, columns=feature_names, index=X.index)
                X_encoded = X_encoded.drop(columns=[col])
                X_encoded = pd.concat([X_encoded, encoded_df], axis=1)
        
        return X_encoded


@dataclass
class ModelConfig:
    model_type: str
    hyperparameters: Dict[str, Any]
    random_state: int = 42


class ModelTrainer(BaseOperation):
    def __init__(self, config: ModelConfig, name: str = None):
        super().__init__(config.__dict__ if hasattr(config, '__dict__') else config, name)
        self.model_config = config if isinstance(config, ModelConfig) else ModelConfig(**config)
        self.model = None
    
    def _initialize_model(self):
        model_type = self.model_config.model_type
        params = self.model_config.hyperparameters.copy()
        params['random_state'] = self.model_config.random_state
        
        if model_type == 'random_forest':
            return RandomForestClassifier(**params)
        elif model_type == 'gradient_boosting':
            return GradientBoostingClassifier(**params)
        elif model_type == 'logistic_regression':
            return LogisticRegression(**params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def execute(self, X_train, y_train, X_val=None, y_val=None) -> Any:
        self.model = self._initialize_model()
        
        print(f"Training {self.model_config.model_type}...")
        
        self.model.fit(X_train, y_train)
        
        train_preds = self.model.predict(X_train)
        train_acc = accuracy_score(y_train, train_preds)
        
        metrics = {
            'train_accuracy': train_acc
        }
        
        if X_val is not None and y_val is not None:
            val_preds = self.model.predict(X_val)
            val_acc = accuracy_score(y_val, val_preds)
            val_f1 = f1_score(y_val, val_preds, average='weighted')
            val_precision = precision_score(y_val, val_preds, average='weighted', zero_division=0)
            val_recall = recall_score(y_val, val_preds, average='weighted', zero_division=0)
            
            metrics.update({
                'val_accuracy': val_acc,
                'val_f1': val_f1,
                'val_precision': val_precision,
                'val_recall': val_recall
            })
        
        self.metadata = metrics
        
        print(f"Training complete: Train Acc={train_acc:.4f}")
        if 'val_accuracy' in metrics:
            print(f"   Validation Acc={metrics['val_accuracy']:.4f}, F1={metrics['val_f1']:.4f}")
        
        return self.model


class HyperparameterTuner(BaseOperation):
    def execute(self, model, X_train, y_train, param_grid: Dict) -> Dict[str, Any]:
        cv = self.config.get('cv', 5)
        scoring = self.config.get('scoring', 'accuracy')
        
        print(f"Tuning hyperparameters with {cv}-fold CV...")
        
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        results = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'best_estimator': grid_search.best_estimator_
        }
        
        self.metadata = {
            'best_score': grid_search.best_score_,
            'n_param_combinations': len(grid_search.cv_results_['params'])
        }
        
        self.save_artifact(results['best_params'], f"best_params_{self.timestamp}.json")
        
        print(f"Best score: {grid_search.best_score_:.4f}")
        print(f"   Best params: {grid_search.best_params_}")
        
        return results


class ModelEvaluator(BaseOperation):
    def execute(self, model, X_test, y_test, task_type: str = 'classification') -> Dict[str, float]:
        y_pred = model.predict(X_test)
        
        if task_type == 'classification':
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred, average='weighted'),
                'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0)
            }
        elif task_type == 'regression':
            metrics = {
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'r2_score': r2_score(y_test, y_pred)
            }
        else:
            raise ValueError(f"Unknown task type: {task_type}")
        
        self.metadata = metrics
        self.save_artifact(metrics, f"evaluation_metrics_{self.timestamp}.json")
        
        print(f"Evaluation complete:")
        for metric_name, metric_value in metrics.items():
            print(f"   {metric_name}: {metric_value:.4f}")
        
        return metrics


class ExperimentTracker(BaseOperation):
    def __init__(self, config: Dict[str, Any], name: str = None):
        super().__init__(config, name)
        self.run_id = None
    
    def execute(self, experiment_name: str = None) -> str:
        experiment_name = experiment_name or self.config.get('experiment_name', 'default_experiment')
        
        mlflow.set_experiment(experiment_name)
        mlflow.start_run(run_name=self.config.get('run_name'))
        
        self.run_id = mlflow.active_run().info.run_id
        
        mlflow.log_params(self.config)
        
        print(f"Experiment started: {experiment_name} (Run ID: {self.run_id})")
        return self.run_id
    
    def end(self):
        mlflow.end_run()
        print(f"Experiment ended: Run ID {self.run_id}")


class ModelRegistry(BaseOperation):
    def execute(self, model, model_name: str, metrics: Dict = None) -> str:
        model_uri = mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name=model_name
        ).model_uri
        
        if metrics:
            mlflow.log_metrics(metrics)
        
        tags = self.config.get('tags', {})
        for key, value in tags.items():
            mlflow.set_tag(key, value)
        
        print(f"Model registered: {model_name}")
        print(f"   Model URI: {model_uri}")
        
        return model_uri


class LocalDeployer(BaseOperation):
    def execute(self, model, model_name: str) -> str:
        import joblib
        
        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)
        
        model_path = model_dir / f"{model_name}_{self.timestamp}.pkl"
        joblib.dump(model, model_path)
        
        inference_script = f"""
import joblib
import pandas as pd

model = joblib.load('{model_path}')

def predict(data):
    if isinstance(data, dict):
        data = pd.DataFrame([data])
    return model.predict(data)
"""
        
        script_path = model_dir / f"{model_name}_inference.py"
        with open(script_path, 'w') as f:
            f.write(inference_script)
        
        print(f"Model deployed locally: {model_path}")
        print(f"   Inference script: {script_path}")
        
        return str(model_path)


class DockerBuilder(BaseOperation):
    def execute(self, model_path: str, app_name: str) -> str:
        dockerfile_content = f"""
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY {model_path} ./model.pkl
COPY app.py .

EXPOSE 8000

CMD ["python", "app.py"]
"""
        
        with open("Dockerfile", 'w') as f:
            f.write(dockerfile_content)
        
        app_content = """
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load('model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
"""
        
        with open("app.py", 'w') as f:
            f.write(app_content)
        
        if self.config.get('build_image', False):
            try:
                client = docker.from_env()
                image, logs = client.images.build(path=".", tag=f"{app_name}:latest")
                print(f"Docker image built: {app_name}:latest")
                return f"{app_name}:latest"
            except Exception as e:
                print(f"Docker build failed: {e}")
                return "Dockerfile created (build manually)"
        else:
            print(f"Dockerfile created: ./Dockerfile")
            return "Dockerfile"


class AWSDeployer(BaseOperation):
    def execute(self, model_uri: str, endpoint_name: str) -> str:
        print(f"AWS deployment stub")
        print(f"   Model URI: {model_uri}")
        print(f"   Endpoint: {endpoint_name}")
        print(f"   Note: Requires AWS credentials and SageMaker setup")
        
        deployment_info = {
            'status': 'stub',
            'model_uri': model_uri,
            'endpoint_name': endpoint_name,
            'deployment_time': self.timestamp
        }
        
        self.save_artifact(deployment_info, f"aws_deployment_{self.timestamp}.json")
        
        return "AWS deployment stub - configure credentials"


class MLPipeline:
    def __init__(self, operations: List[BaseOperation], name: str = "ml_pipeline"):
        self.operations = operations
        self.name = name
        self.results = {}
        self.experiment_tracker = None
    
    def run(self, initial_input: Any = None, experiment_name: str = None) -> Dict[str, Any]:
        print(f"\n{'='*60}")
        print(f"Starting Pipeline: {self.name}")
        print(f"{'='*60}\n")
        
        if experiment_name:
            self.experiment_tracker = ExperimentTracker({'experiment_name': experiment_name})
            self.experiment_tracker.execute()
        
        result = initial_input
        
        for i, operation in enumerate(self.operations, 1):
            print(f"\n[{i}/{len(self.operations)}] Running {operation.name}...")
            print("-" * 60)
            
            try:
                if isinstance(result, dict):
                    result = operation.execute(**result)
                else:
                    result = operation.execute(result)
                
                self.results[operation.name] = result
                
                if self.experiment_tracker:
                    operation.log_to_mlflow()
                
            except Exception as e:
                print(f"Error in {operation.name}: {e}")
                raise
        
        if self.experiment_tracker:
            self.experiment_tracker.end()
        
        print(f"\n{'='*60}")
        print(f"Pipeline Complete: {self.name}")
        print(f"{'='*60}\n")
        
        return self.results


if __name__ == "__main__":
    print("\nMLOps Framework - Unified Pipeline System\n")
    print("Framework loaded successfully!")
    print("   Use example_classification_pipeline() to run full demo")
    print("   Or use quick_train_model() for fast training")