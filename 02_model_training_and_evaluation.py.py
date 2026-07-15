import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
data=pd.read_csv("heart data.csv")
df=pd.DataFrame(data)

# first 5 rows
print(df.head())

# information
print(df.info())

# missing values
print(df.isnull().sum())

# Statistical Summary
print(df.describe())

# Dataset Shape
print("Shape :", df.shape)

# Column Names
print(df.columns)

df["Cholesterol"]=df["Cholesterol"].replace(0,np.nan)
df["Cholesterol"]=df["Cholesterol"].fillna(df["Cholesterol"].median())

print(df.isnull().sum())

# ===============
# Encoding
# ===============
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

# Binary Encoding
df["Sex"] = le.fit_transform(df["Sex"])
df["ExerciseAngina"] = le.fit_transform(df["ExerciseAngina"])

# One-Hot Encoding
df = pd.get_dummies(df,columns=["ChestPainType", "RestingECG", "ST_Slope"],dtype=int)

print(df.head())

# ==========================================================
# Features & Target
# ==========================================================

X=df.drop("HeartDisease",axis=1)
y=df["HeartDisease"]

print("\nFeatures shape: ",X.shape)
print("\nTarget shape: ",y.shape)

# ==========================================================
# Train Test Split
# ==========================================================
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

print("Training Features :", X_train.shape)
print("Testing Features :", X_test.shape)

print("Training Target :", y_train.shape)
print("Testing Target :", y_test.shape)


# ============================
# Feature scaling
# ============================
numeric_columns=["Age","RestingBP","Cholesterol","MaxHR","Oldpeak"]

from sklearn.preprocessing import StandardScaler
std_scaler=StandardScaler()
X_train[numeric_columns]=std_scaler.fit_transform(X_train[numeric_columns])
X_test[numeric_columns]=std_scaler.transform(X_test[numeric_columns])

print("\nScaling Completed Successfully.")

print(X_train.head())
print(X_test.head())

# ============================
#  ML models
# ============================

models={
    "Logistic Regression":LogisticRegression(max_iter=1000),
    "Decision Trees":DecisionTreeClassifier(random_state=42),
    "Random Forest":RandomForestClassifier(random_state=42),
    "KNN":KNeighborsClassifier(),
    "Naive Bayes":GaussianNB(),
    "SVM":SVC()
}

# ===========================
# Model Training
# ===========================
results=[]

for model_name,model in models.items():
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)

    accuracy=accuracy_score(y_test,y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results.append([model_name,accuracy,precision,recall,f1])

    print(f"{model_name} completed successfully")


# ==========================================================
# Model Comparison
# ==========================================================

comparison = pd.DataFrame(results,columns=["Model","Accuracy","Percision","Recall","F1"])

comparison = comparison.sort_values(by="Accuracy", ascending=False)
print("\nModel Performance Comparison\n")
print(comparison.round(3))

from sklearn.model_selection import cross_val_score
cv_results=[]


# =============================
# CROSS VALIDATION
# =============================

for model_name,model in models.items():
    scores=cross_val_score(model,X_train,y_train,cv=5,scoring="accuracy")
    cv_results.append([model_name,scores.mean(),scores.std()])  
    
# Comparision
cv_comparison=pd.DataFrame(cv_results,columns=["Model name","CV accuracy","CV std"])
cv_comparison=cv_comparison.sort_values(by="CV accuracy",ascending=False)

print(f"\nCross Validation results:=\n{cv_comparison.round(3)}")


# VISUALIZATION

plt.figure(figsize=(10,5))
plt.bar(comparison["Model"],comparison["Accuracy"],color="steelblue",edgecolor="black")
plt.title("Model accuracy comparision",fontsize=15,fontweight="bold",color="maroon")
plt.xlabel("Models",color="maroon",fontsize=12)
plt.ylabel("Accuracy",color="maroon",fontsize=12)
plt.tight_layout()
plt.xticks()
plt.grid(axis="y",alpha=0.3)
plt.show()

'''  **** CROSS VALIDATION CISUALIZATION ****   '''
plt.figure(figsize=(10,5))

plt.bar(cv_comparison["Model name"],cv_comparison["CV accuracy"],color="crimson",edgecolor="black")
plt.title("Cross Validation Accuracy Comparison", fontsize=15, fontweight="bold",color="blue")
plt.xlabel("Models",color="green",fontsize=12)
plt.ylabel("CV Accuracy",color="green",fontsize=12)
plt.xticks()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()



import joblib
joblib.dump(models["SVM"],"heart_disease_model.pkl")
joblib.dump(std_scaler,"scaler.pkl")

model = joblib.load("heart_disease_model.pkl")
scaler = joblib.load("scaler.pkl")
print(model)
print(scaler)
joblib.dump(X.columns.tolist(), "heart_columns.pkl")
