import pandas as pd
df = pd.read_csv("C:/Users/User/OneDrive/Documents/data.csv.csv")
print(df.head())

df = df.drop(['RowNumber','Surname'], axis=1)

df.columns = ['CustomerId','CreditScore','Geography','Gender','Age','Tenure',
              'Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary','Exited']

df['Gender'] = df['Gender'].map({'Female':0, 'Male':1})

print(df.head())
print(df.info())

print(df.isnull().sum())    # counting the missing values
print(df.duplicated().sum())   #check duplicates
print(df.describe())   # gives statstical analysis for all columns


numeric_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
print(df[numeric_features].describe())
#gives statstical analysis for mentioned columns

import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x='Exited', data=df)
plt.title("Churn Distribution")
plt.show()
#shows how many customers churned (Exited=1) vs stayed (Exited=0)

numeric_features = ['CreditScore','Age','Tenure','Balance','NumOfProducts','EstimatedSalary']
df[numeric_features].hist(bins=15, figsize=(15,10))
plt.show()
#count vs different columns

df.groupby('Exited')[['CreditScore','Age','Balance','EstimatedSalary']].mean()  
#basic stats summary

print(df.columns.tolist())  #shows all columns

import seaborn as sns              #analyse exit patterns on basis of geography
import matplotlib.pyplot as plt
sns.countplot(x='Geography', hue='Exited', data=df)
plt.title("Churn by Geography")
plt.show()

df['AgeGroup'] = pd.cut(df['Age'], bins=[18,30,40,50,60,100], labels=['18-30','30-40','40-50','50-60','60+'])
sns.countplot(x='AgeGroup', hue='Exited', data=df)
plt.title("Churn by Age Group")
plt.show()

corr = df.corr(numeric_only=True)                            # heatmap
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt

# Drop ID + target column
X = df.drop(columns=['CustomerId','Exited'])

# Convert categorical to numeric
X = pd.get_dummies(X, drop_first=True)

# Target
y = df['Exited']

# Train simple model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Feature importances
importances = pd.Series(model.feature_importances_, index=X.columns)
importances.sort_values(ascending=False).plot(kind='bar', figsize=(10,5))
plt.title("Feature Importance")
plt.show()

sns.catplot(x="IsActiveMember", col="Geography", hue="Exited", kind="count", data=df)       #two features vs churn
#France:Inactive members (IsActiveMember = 0) churn a lot more than active ones.But even active members churn a bit, though less.
#Spain:Very few churners overall. Being active seems to reduce churn further.
#Germany:Highest churn rates compared to France/Spain.Even active members in Germany churn noticeably.


df['Exited'].value_counts(normalize=True)

sns.boxplot(x=df['Balance'])       #outliers

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import seaborn as sns

# Features and Target
X = df.drop(columns=['CustomerId','Exited','AgeGroup'])  # drop ID + derived column
X = pd.get_dummies(X, drop_first=True)  # ensure categorical encoded
y = df['Exited']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Logistic Regression
log_reg = LogisticRegression(max_iter=1000, class_weight='balanced')
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)
y_prob = log_reg.predict_proba(X_test)[:,1]

print("Logistic Regression:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Logistic Regression")
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, label="Logistic (AUC={:.2f})".format(roc_auc_score(y_test, y_prob)))
plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

import sys
print(sys.executable)
#sys acts as the bridge between your ML code and your computer's operating system. 
# In a machine learning workflow, sys is used purely for configuration, automation, and performance checking

#RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight='balanced'
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)
rf_prob = rf.predict_proba(X_test)[:,1]

#XGBOOST
from xgboost import XGBClassifier

xgb = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    random_state=42
)

xgb.fit(X_train, y_train)

xgb_pred = xgb.predict(X_test)
xgb_prob = xgb.predict_proba(X_test)[:,1]

#COMPARISON TABLE
results = pd.DataFrame({

    'Model':[
        'Logistic Regression',
        'Random Forest',
        'XGBoost'
    ],

    'Accuracy':[
        accuracy_score(y_test,y_pred),
        accuracy_score(y_test,rf_pred),
        accuracy_score(y_test,xgb_pred)
    ],

    'Precision':[
        precision_score(y_test,y_pred),
        precision_score(y_test,rf_pred),
        precision_score(y_test,xgb_pred)
    ],

    'Recall':[
        recall_score(y_test,y_pred),
        recall_score(y_test,rf_pred),
        recall_score(y_test,xgb_pred)
    ],

    'F1':[
        f1_score(y_test,y_pred),
        f1_score(y_test,rf_pred),
        f1_score(y_test,xgb_pred)
    ],

    'ROC_AUC':[
        roc_auc_score(y_test,y_prob),
        roc_auc_score(y_test,rf_prob),
        roc_auc_score(y_test,xgb_prob)
    ]
})

print(results.round(3))

#VISUALISE
results.set_index('Model').plot(
    kind='bar',
    figsize=(10,6)
)


importance = pd.Series(
    xgb.feature_importances_,
    index=X_train.columns
)

importance.sort_values(
    ascending=False
).head(10)


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# ── Evaluate Random Forest ──────────────────────────────────────────────
print("Random Forest:")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print("Precision:", precision_score(y_test, rf_pred))
print("Recall:", recall_score(y_test, rf_pred))
print("F1:", f1_score(y_test, rf_pred))
print("ROC-AUC:", roc_auc_score(y_test, rf_prob))

# ── Evaluate XGBoost ─────────────────────────────────────────────────────
xgb_pred = xgb.predict(X_test)
xgb_prob = xgb.predict_proba(X_test)[:,1]

print("\nXGBoost:")
print("Accuracy:", accuracy_score(y_test, xgb_pred))
print("Precision:", precision_score(y_test, xgb_pred))
print("Recall:", recall_score(y_test, xgb_pred))
print("F1:", f1_score(y_test, xgb_pred))
print("ROC-AUC:", roc_auc_score(y_test, xgb_prob))

# ── Model Comparison Table ────────────────────────────────────────────────
results = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest', 'XGBoost'],
    'Accuracy':  [accuracy_score(y_test, y_pred),    accuracy_score(y_test, rf_pred),  accuracy_score(y_test, xgb_pred)],
    'Precision': [precision_score(y_test, y_pred),   precision_score(y_test, rf_pred), precision_score(y_test, xgb_pred)],
    'Recall':    [recall_score(y_test, y_pred),       recall_score(y_test, rf_pred),    recall_score(y_test, xgb_pred)],
    'F1':        [f1_score(y_test, y_pred),           f1_score(y_test, rf_pred),        f1_score(y_test, xgb_pred)],
    'ROC-AUC':   [roc_auc_score(y_test, y_prob),      roc_auc_score(y_test, rf_prob),   roc_auc_score(y_test, xgb_prob)]
})
print("\nModel Comparison:\n", results)

# ── Combined ROC Curve ────────────────────────────────────────────────────
for prob, name in [(y_prob, 'Logistic'), (rf_prob, 'Random Forest'), (xgb_prob, 'XGBoost')]:
    fpr, tpr, _ = roc_curve(y_test, prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, prob):.2f})")
plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison"); plt.legend(); plt.show()

# ── XGBoost: Top Churn Drivers ────────────────────────────────────────────
importances = pd.Series(xgb.feature_importances_, index=X_test.columns)
top_features = importances.sort_values(ascending=False).head(8)
top_features.plot(kind='bar', figsize=(10,5), color='steelblue')
plt.title("XGBoost - Top Churn Drivers"); plt.tight_layout(); plt.show()

# ── Churn Risk Scoring ────────────────────────────────────────────────────
# Assign each customer a churn probability and risk tier
churn_scores = xgb.predict_proba(X_test)[:,1]
risk_df = X_test.copy()
risk_df['ChurnProbability'] = churn_scores
risk_df['RiskTier'] = pd.cut(churn_scores,
                              bins=[0, 0.3, 0.6, 1.0],
                              labels=['Low Risk', 'Medium Risk', 'High Risk'])
print(risk_df['RiskTier'].value_counts())

# ── Retention Strategy Mapping ────────────────────────────────────────────
# Based on XGBoost top drivers: Age, Balance, NumOfProducts, IsActiveMember,
# Geography, CreditScore, Tenure

strategy_map = {
    'High Risk': [
        "Assign dedicated relationship manager",
        "Offer personalised retention discount or loyalty reward",
        "Proactive outreach call within 7 days",
        "Upgrade product bundle (NumOfProducts driver)",
        "Re-engagement campaign for inactive members"
    ],
    'Medium Risk': [
        "Send targeted email: highlight unused features/benefits",
        "Offer flexible plan downgrade to reduce Balance-driven stress",
        "Germany-specific: regional trust-building campaigns (high churn geography)",
        "Prompt to activate membership benefits (IsActiveMember driver)"
    ],
    'Low Risk': [
        "Standard loyalty programme communications",
        "Cross-sell second product (NumOfProducts is a key driver)",
        "Annual satisfaction survey"
    ]
}

print("\n── RETENTION STRATEGIES BY RISK TIER ──")
for tier, strategies in strategy_map.items():
    print(f"\n{tier}:")
    for s in strategies:
        print(f"  → {s}")

# ── Strategy Distribution Plot ────────────────────────────────────────────
risk_counts = risk_df['RiskTier'].value_counts()
colors = ['#d62728', '#ff7f0e', '#2ca02c']
risk_counts.plot(kind='bar', color=colors, figsize=(7,4))
plt.title("Customer Distribution by Churn Risk Tier")
plt.ylabel("Number of Customers"); plt.xticks(rotation=0); plt.tight_layout(); plt.show()


# ── Feature Importance Comparison: LR vs XGBoost ─────────────────────────
import numpy as np

# Logistic Regression coefficients (absolute value = importance)
lr_importance = pd.Series(np.abs(log_reg.coef_[0]), index=X_test.columns)
lr_top = lr_importance.sort_values(ascending=False).head(8)

# XGBoost importances
xgb_top = pd.Series(xgb.feature_importances_, index=X_test.columns)\
            .sort_values(ascending=False).head(8)

# Plot side by side
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

lr_top.plot(kind='bar', ax=axes[0], color='coral')
axes[0].set_title("Top Churn Drivers - Logistic Regression")
axes[0].set_ylabel("Coefficient Magnitude")
axes[0].tick_params(axis='x', rotation=45)

xgb_top.plot(kind='bar', ax=axes[1], color='steelblue')
axes[1].set_title("Top Churn Drivers - XGBoost")
axes[1].set_ylabel("Feature Importance Score")
axes[1].tick_params(axis='x', rotation=45)

plt.suptitle("Why Models Disagree: Linear vs Non-Linear Feature Importance", fontsize=13)
plt.tight_layout()
plt.show()