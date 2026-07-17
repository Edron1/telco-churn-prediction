import logging

from model_params import ModelParams, EncoderParams
from visualizer import Visualizer
from data_processor import DataProcessor
from config import DataConfig
from model_trainer import ModelTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('churn_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    logger.info("Stating churn analysis")
    
    model_params = ModelParams()
    encoder_params = EncoderParams()
    data_config = DataConfig()

    processor = DataProcessor(data_config)
    processor.load_data('data/WA_Fn-UseC_-Telco-Customer-Churn.csv').preprocess()
    X, y = processor.get_data()
    data = processor.get_full_data()

    logger.info(f"Data loaded: {data.shape}")
    logger.info(f"Distribution of the target variable:\n {y.value_counts(normalize=True)}")

    visualizer = Visualizer("graphs")

    visualizer.countplot(
        data=data,
        x='gender',
        hue='Churn',
        title="Distribution of outflow by gender",
        xlabel='Gender',
        ylabel='Number of clients',
        save_path='churn_by_gender',
        palette=['#3498db', '#e74c3c'],
        figsize=(10, 4)
    )

    visualizer.countplot(
        data=data,
        x='Contract',
        hue='Churn',
        title='Outflow by contract type',
        xlabel='Contract type',
        ylabel='Number of clients',
        save_path='churn_by_contract',
        palette=['#2ecc71', '#e74c3c'],
        figsize=(10, 4)
    )

    numeric_cols = X.select_dtypes(include=['float64', 'int64'])
    correlation_matrix = numeric_cols.corr()
    visualizer.heatmap(
        data=correlation_matrix,
        annot=True,
        cmap='coolwarm',
        fmt='.2f',
        linewidths=0.5,
        title='Correlation matrix of numerical features',
        save_path='correlation_heat_map',
        figsize=(10, 8)
    )

    trainer = ModelTrainer(data_config, model_params, encoder_params)
    results = trainer.train_all_models(X, y)

    trainer.print_results()

    best_model_type = trainer.best_model
    logger.info(f"\n BEST MODEL: {best_model_type}")

    importances_df = trainer.get_feature_importance(best_model_type)
    top_10 = importances_df.head(10)

    logger.info("Top 10 importance fetures:")
    logger.info(top_10)

    visualizer.barplot(
        data=top_10,
        x='importance',
        y='feature',
        title=f'Top 10 importance fetures ({best_model_type})',
        xlabel='Importance',
        ylabel='Feature',
        save_path=f'top10_features_{best_model_type}',
        palette='viridis',
        figsize=(10, 6)
    )

    trainer.save_model(best_model_type)

    logger.info("Analysis completed")

    return trainer, results

if __name__ == "__main__":
    trainer, results = main()
