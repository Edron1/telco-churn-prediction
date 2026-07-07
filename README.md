# Telco Customer Churn Prediction

## 📌 Project description
**Task:** Predict whether a customer will terminate their contract with a telecom company in the near future.

**Business-value:** Early identification of customers who are likely to churn allows a company to offer them discounts or improve their service while maintaining revenue.

## 📊 Data
Dataset used **Telco Customer Churn** from IBM. It contains information about 7,043 customers and 21 attributes: demographics, services, contract type, payment method, and service duration.

## 🛠️ Tools used
- **Pandas** — data loading and preprocessing
- **Scikit-learn** — pipeline construction, One-Hot Encoding, models
- **XGBoost** — gradient boosting
- **Matplotlib / Seaborn** — visualization

## 🔍 Key steps
1. **Preprocessing:** transformation `TotalCharges` to numeric format, filling in the blanks.
2. **Coding of categorical features** through the `OneHotEncoder` in the pipeline.
3. **Training two models:** RandomForest and XGBoost.
4. **Cross-validation** to assess stability.
5. **Analysis of the importance of signs** (Feature Importance).
6. **Visualization of key dependencies:** outflow by contract type, gender, and correlation.

## 📈 Results

| Model        | Accuracy | Cross-validated Accuracy |
|--------------|----------|--------------------------|
| RandomForest |  ~74.8%  |         ~75.3%           |
| XGBoost      |  ~74.8%  |         ~75.1%           |

**Top 3 factors affecting outflow:**
1. Duration of the contract (`Contract`) — Customers with monthly contracts leave more often.
2. Lack of service `OnlineSecurity`.
3. Duration of service (`tenure`) — the longer a customer stays, the less likely they are to leave.

## 📷 Visualization

| Chart                               | Description                                     |
|-------------------------------------|-------------------------------------------------|
| `top10_most_important_features.png` | Top 10 Signs That Are Important for RandomForest|
| `churn_by_contract.png`             | Distribution of outflow by type of contract     |
| `churn_by_gender.png`               | Outflow distribution by gender                  |
| `correlation_heat_map.png`          | Correlations of numerical features              |

## 🚀 How to launch a project
1. Clone the repository:
   ```bash
   git clone https://github.com/Edron1/telco-churn-prediction.git
2. Install the dependencies:
   pip install -r requirements.txt
3. Run the script:
   python churn_analysis.py