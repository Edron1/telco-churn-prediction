from dataclasses import dataclass

@dataclass
class DataConfig:
    target_column: str = "Churn"
    id_column: str = "customerID"
    categorical_columns: list = None
    numerical_columns: list = None
    dropna_threshold: float = 0.5

    def __post_init__(self):
        if self.categorical_columns is None:
            self.categorical_columns = [
                'gender', 'SeniorCitizen', 'Partner', 'Dependents',
                'PhoneService', 'MultipleLines', 'InternetService',
                'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport', 'StreamingTV', 'StreamingMovies',
                'Contract', 'PaperlessBilling', 'PaymentMethod'
            ]
        if self.numerical_columns is None:
            self.numerical_columns = [
                'tenure', 'MonthlyCharges', 'TotalCharges'
            ]