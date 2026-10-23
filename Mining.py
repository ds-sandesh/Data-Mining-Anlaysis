# Data Mining from Scratch 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Load the dataset
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
print(df.head())
print('Data loaded successfully! \n Data shape:', df.shape)

#Information about the dataset and EDA
print(df.info())
print(df.describe())
print(df.isnull().sum())
df['Churn'].value_counts(normalize = True)

#Fixing the data types issues
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

#For new customers, TotalCharges is missing. Let's check the tenure and MonthlyCharges for those customers.
df[df['TotalCharges'].isnull()][['tenure', 'MonthlyCharges']].head()
df['TotalCharges'] = df['TotalCharges'].fillna(0)

#Drop customerID column as it is not useful for analysis
df = df.drop('customerID', axis=1)

#Numercal distribution of the dataset
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col in zip(axes, ['tenure', 'MonthlyCharges', 'TotalCharges']):
    sns.histplot(df[col], kde=True, ax=ax)
    ax.set_title(f'Distribution of {col}')
plt.tight_layout()
plt.show()

#Target variable distribution
sns.countplot(x='Churn', data=df)

#Bivariate analysis of categorical features with respect to the target variable
#Categorical  vs churn
for col in ['Contract', 'InternetService', 'PaymentMethod']:
    ct = pd.crosstab(df[col], df['Churn'], normalize='index')
    print(ct, '\n')
    ct.plot(kind='bar', stacked=True)
    plt.title(f"{col} vs Churn")
    plt.show()

#Numerical vs churn
for col in ['tenure', 'MonthlyCharges', 'TotalCharges']:
    sns.boxplot(x='Churn', y=col, data=df)
    plt.title(f"{col} vs Churn")
    plt.show()

#Encode target temporarily for correlation analysis
df_corr = df.copy()
df_corr['Churn'] = df_corr['Churn'].map({'No': 0, 'Yes': 1})
df_corr = pd.get_dummies(df_corr)

corr = df_corr.corr()['Churn'].sort_values(ascending=False)
print(corr.head(15))

#Feature Engineering
#Average monthly spend(accouts for discounts over time)
df['AvgMonthlySpend'] = df['TotalCharges'] / (df['tenure'].replace(0, 1))  # Avoid division by zero

#Number of services subscribed by the customer
service_cols = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
df['NumServices'] = df[service_cols].apply(lambda x: sum(x == 'Yes'), axis=1)

#Is customer new?
df['IsNewCustomer'] = df['tenure'].apply(lambda x: 1 if x <= 12 else 0).astype(int)

#Highvalue customer (paying above median)
df['IsHighValueCustomer'] = (df['MonthlyCharges'] > df['MonthlyCharges'].median()).astype(int)

#Encoding categorical variables
#Binaary encoding for Yes/No columns
binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
for col in binary_cols:
    df[col] = (df[col] == 'Yes').astype(int)

#target Encoding
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

#Multi category: One hot encoding
df = pd.get_dummies(df, drop_first=True)  #drop_first to avoid dummy variable trap (Multicollinearity)

#Splitting the dataset into training and testing sets
from sklearn.model_selection import StratifiedKFold, train_test_split

X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#Handling class imbalance using SMOTE
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)



#Scaling data using StandardScaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) #Note: transform only, not fit_transform to avoid data leakage

#Building a Logistic Regression model
from sklearn.dummy import DummyClassifier
baseline = DummyClassifier(strategy='most_frequent')
baseline.fit(X_train_scaled, y_train)
print(baseline.score(X_test_scaled, y_test)) 

#Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, roc_auc_score

lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
lr.fit(X_train_scaled, y_train)
preds = lr.predict(X_test_scaled)
probs = lr.predict_proba(X_test_scaled)[:, 1]

print(classification_report(y_test, preds))
print('ROC AUC Score:', roc_auc_score(y_test, probs))

#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=300, max_depth=10,
                            class_weight='balanced', random_state=42, n_jobs=-1)
rf.fit(X_train_scaled, y_train)

#gradient boosting classifier
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.5, max_depth=4, random_state=42)
gb.fit(X_train_scaled, y_train)

#Cross-validation for model evaluation
from sklearn.model_selection import cross_val_score, StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(rf, X_train_scaled, y_train, cv=cv, scoring='roc_auc')
print(f"AUC: {scores.mean():.3f} ± {scores.std():.3f}")

#hyperparameter tuning using GridSearchCV
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 300],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5, 10]
}

grid = GridSearchCV(rf, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1)
grid.fit(X_train_scaled, y_train)
print("Best parameters found: ", grid.best_params_)

#Confusion Matrix:
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, preds)
ConfusionMatrixDisplay(cm).plot()

#ROC-AUC vs PR-AUC
from sklearn.metrics import roc_curve, precision_recall_curve, auc

#roc curve
fpr, tpr, thresholds = roc_curve(y_test, probs)
plt.plot(fpr, tpr, label=f'ROC curve (area = {auc(fpr, tpr):.2f})')

#PR - curve
precision, recall, thresholds = precision_recall_curve(y_test, probs)
plt.plot(recall, precision)

#Threshold tuning for optimal F1 score
thresholds = np.arange(0.1, 0.9, 0.05)
for t in thresholds:
    preds_t = (probs >= t)
    from sklearn import metrics
print(f"t = {t:.2f} | precision: {metrics.precision_score(y_test, preds_t):.3f} "
      f"| recall: {metrics.recall_score(y_test, preds_t):.3f}")

#Model interpretation
#feature importance
importances = pd.Series(rf.feature_importances_, index=X.columns)
importances.sort_index().tail(10).plot(kind='barh')

#SHAP values for model interpretation
import shap
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)

#Deployment Thinking
#Serializing the model using joblib
import joblib
joblib.dump(rf, 'churn_model.pkl')
joblib.dump(scaler, 'scaler.pkl')