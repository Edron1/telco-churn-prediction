# Telco Customer Churn Prediction

## 📌 Project description
**Task:** Predict whether a customer will terminate their contract with a telecom company in the near future.

**Business-value:** Early identification of customers who are likely to churn allows a company to offer them discounts or improve their service while maintaining revenue.

## 📊 Data
Dataset used **Telco Customer Churn** from IBM. It contains information about 7,043 customers and 21 attributes: demographics, services, contract type, payment method, and service duration.

## 🛠️ Tech stack

### Core Libraries
- **Pandas** - Data loading, preprocessing, and manipulation
- **NumPy** - Numerical operations and array handling
- **Scikit-learn** - Pipeline construction, encoding, cross-validation
- **XGBoost** - Gradient boosting implementation
- **Matplotlib/Seaborn** - Data visualization and plotting
- **Joblib** - Model serialization and saving

### Development Tools
- **Python 3.8+**
- **Logging** - Structured logging for debugging and monitoring
- **Git** - Version control
- **VS Code** - Development environment

## 🔍 Key steps
1. **Preprocessing:** transformation `TotalCharges` to numeric format, filling in the blanks.
2. **Coding of categorical features** through the `OneHotEncoder` in the pipeline.
3. **Training two models:** RandomForest and XGBoost.
4. **Cross-validation** to assess stability.
5. **Analysis of the importance of signs** (Feature Importance).
6. **Visualization of key dependencies:** outflow by contract type, gender, and correlation.

## 📈 Results

| Model        | Accuracy  | Cross-validated Accuracy  |
|--------------|-----------|---------------------------|
| RandomForest |  ~77.57%  |         ~78.75%           |
| XGBoost      |  ~76.08%  |         ~77.88%           |

**Top 3 factors affecting outflow:**
1. Duration of the contract (`Contract`) — Customers with monthly contracts leave more often.
2. Lack of service `OnlineSecurity`.
3. Duration of service (`tenure`) — the longer a customer stays, the less likely they are to leave.

## 📷 Visualization

| Chart                               | Description                                               |
|-------------------------------------|-----------------------------------------------------------|
| `top10_features_model_type.png`     | Top 10 features in the models that showed the best results|
| `churn_by_contract.png`             | Distribution of outflow by type of contract               |
| `churn_by_gender.png`               | Outflow distribution by gender                            |
| `correlation_heat_map.png`          | Correlations of numerical features                        |

## Project architecture
telco-churn-prediction/
│
├── data/
│
├── models/
│
├── graphs/
│
├── churn_analysis.py
├── data_processor.py
├── model_trainer.py
├── visualizer.py
├── config.py
├── model_params.py
└── requirements.txt

## 🚀 How to launch a project
1. Clone the repository:
   ```bash
   git clone https://github.com/Edron1/telco-churn-prediction.git
   cd telco-churn-prediction
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
4. Run the script:
   ```bash
   python churn_analysis.py