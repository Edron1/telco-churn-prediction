import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import seaborn as sns

data = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data['TotalCharges'] = data['TotalCharges'].fillna(data['TotalCharges'].median())

print(data.head())
print(data.info())
print(data.describe())

X = data.drop(columns=['Churn', 'customerID'])
y = (data['Churn'] == 'Yes').astype(int)

categorical_cols = X.select_dtypes(include=['object', 'str']).columns.to_list()

preprocessor = ColumnTransformer([
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model_1_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=0, n_estimators=50))
])

model_1_pipeline.fit(X_train, y_train)
pred_1 = model_1_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, pred_1)

print(f"Accuracy: {accuracy*100:.2f}%")

scores = cross_val_score(model_1_pipeline, X_train, y_train, cv=5)
print(f'Cross-validation mean accuracy: {scores.mean():.4f}')

model_2_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(random_state=0, n_estimators=100, learning_rate=0.5))
])

model_2_pipeline.fit(X_train, y_train)

pred_2 = model_2_pipeline.predict(X_test)

accuracy = accuracy_score(y_test, pred_2)

print(f"Accuracy: {accuracy*100:.2f}%")

scores = cross_val_score(model_2_pipeline, X_train, y_train, cv=5)
print(f'Cross-validation mean accuracy: {scores.mean():.4f}')

print(y.value_counts(normalize=True)) #73% No, 26% Yes

feature_names = preprocessor.get_feature_names_out()
model_features = model_1_pipeline.named_steps['classifier']
importances = model_features.feature_importances_

feature_imporances_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
})

top_10 = feature_imporances_df.sort_values('importance', ascending=False).head(10)
print(top_10)

#Statistical chart for clients
sns.countplot(data=data, x='gender', hue='Churn', palette=['skyblue', 'salmon'])
plt.title('Statistical graph of customer churn')
plt.xlabel('Gender')
plt.ylabel('Clients count')
plt.legend(title="Churn", labels=['No', 'Yes'])
plt.show()
plt.savefig('graphs/churn_by_gender')
#Top 10 most important features
plt.figure(figsize=(10, 6))
sns.barplot(data=top_10, x='importance', y='feature', palette='viridis')
plt.title('Top 10 most important features (RandomForest)')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()
plt.savefig('graphs/top10_most_important_features')
#Churn for contract
sns.countplot(data=data, x='Contract', hue='Churn', palette=['skyblue', 'salmon'])
plt.title('Churn for contract')
plt.xlabel('Contract')
plt.ylabel('Churn')
plt.legend(title="Churn", labels=['No', 'Yes'])
plt.show()
plt.savefig('graphs/churn_by_contract')
#Correlation heat map
numeric_cols = X.select_dtypes(include=['float64', 'int64'])
correlation_matrix = numeric_cols.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(data=correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation heat map')
plt.show()
plt.savefig('graphs/correlation_heat_map')