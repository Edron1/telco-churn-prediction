import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import logging
from typing import Dict, Any, Tuple
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, config, model_params, encoder_params):
        self.config = config
        self.model_params = model_params
        self.encoder_params = encoder_params
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_score = 0
    
    def create_preprocessor(self, categorical_cols: list) -> ColumnTransformer:
        return ColumnTransformer([
            ('onehot', OneHotEncoder(
                handle_unknown=self.encoder_params.handle_unknown, 
                sparse_output=self.encoder_params.sparse_output
            ), categorical_cols)
        ])
    
    def split_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.model_params.test_size, 
            random_state=self.model_params.random_state,
            stratify=y
        )

        logger.info(f"Data separation: train={len(X_train)}, test={len(X_test)}")

        return X_train, X_test, y_train, y_test
    
    def create_model(self, model_type: str, preprocessor: ColumnTransformer) -> Pipeline:
        if model_type == 'random_forest':
            classifier = RandomForestClassifier(
                random_state=self.model_params.random_state,
                n_estimators=self.model_params.n_estimators,
                max_depth=self.model_params.max_depth,
                min_samples_split=self.model_params.min_samples_split,
                n_jobs=-1
            )
        elif model_type == 'xgboost':
            classifier = XGBClassifier(
                random_state=self.model_params.random_state,
                n_estimators=self.model_params.n_estimators,
                learning_rate=self.model_params.learning_rate,
                max_depth=self.model_params.max_depth,
                use_label_encoder=False,
                eval_metric='logloss'
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', classifier)
        ])
    
    def train_model(self, model_type: str, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        logger.info(f"Train model {model_type}")    

        categorical_cols = X_train.select_dtypes(include=['object', 'str']).columns.to_list()

        preprocessor =  self.create_preprocessor(categorical_cols)

        model = self.create_model(model_type, preprocessor)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:,-1]

        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=self.model_params.cv,
            scoring='accuracy'
        )
        
        results = {
            'model_type': model_type,
            'model': model,
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'roc_auc': roc_auc,
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }

        self.models[model_type] =  model
        self.results[model_type] = results

        if roc_auc > self.best_score:
            self.best_score = roc_auc
            self.best_model = model_type

        logger.info(f"Model {model_type}: accuracy = {accuracy:.4f}, cv = {cv_scores.mean():.4f}, roc-auc-score = {roc_auc:.4f}")

        return results
    
    def train_all_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Dict]:
        X_train, X_test, y_train, y_test =  self.split_data(X, y)

        model_types = ['random_forest', 'xgboost']

        for model in model_types:
            self.train_model(model, X_train, y_train, X_test, y_test)
        
        return self.results

    def get_feature_importance(self, model_type: str) -> pd.DataFrame:
        if model_type not in self.models:
            raise ValueError(f"Model {model_type} not trained")
        
        model = self.models[model_type]

        preprocessor = model.named_steps['preprocessor']
        feature_names_encoded = preprocessor.get_feature_names_out()

        classifier = model.named_steps['classifier']
        if hasattr(classifier, 'feature_importances_'):
            importances = classifier.feature_importances_
        else:
            raise ValueError(f"Model {model_type} doesn't have feature_importances_")
        
        importances_df = pd.DataFrame({
            'feature': feature_names_encoded,
            'importance': importances
        }).sort_values('importance', ascending=False)

        return importances_df

    def save_model(self, model_type: str, path: str = "models/"):
        if model_type not in self.models:
            raise ValueError(f"Model {model_type} not trained")
        
        Path(path).mkdir(parents=True, exist_ok=True)
        model_path = f"{path}/{model_type}.joblib"
        joblib.dump(self.models[model_type], model_path)
        logger.info(f"Model saved in {model_path}")
    
    def load_model(self, model_type: str, path: str = "models/"):
        model_path = f"{path}/{model_type}.joblib"
        self.models[model_type] = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
        return self.models[model_type]

    def print_results(self):
        print("\n" + "="*60)
        print("MODEL TRAINING RESULTS")
        print("="*60)
        
        for model_type, results in self.results.items():
            print(f"\n📊 {model_type.upper()}")
            print(f"   Accuracy: {results['accuracy']:.4f}")
            print(f"   CV Mean:  {results['cv_mean']:.4f} (±{results['cv_std']:.4f})")
            print(f"   {'ROC-AUC':<15}: {results['roc_auc']:.4f}")
            print(f"\n   Classification Report:")
            print(results['classification_report'])
            print(f"   Confusion Matrix:")
            print(results['confusion_matrix'])