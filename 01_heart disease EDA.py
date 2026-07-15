import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

'''********  EDA  *******'''

# how many patients have heart disease
heart_patients=df["HeartDisease"].value_counts()
print(heart_patients)
plt.figure(figsize=(5,5))
plt.bar(heart_patients.index,heart_patients.values,color=['orange','red'])
plt.title("Heart patient distribution",fontsize=15,color="blue")
plt.xlabel("Heart disease",fontsize=15,color="blue")
plt.ylabel("No. of patients",fontsize=15,color="blue")
plt.xticks([0,1],["No disease","Heart disease"])
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# How many heart disease cases are present in each gender?


# total patients by gender
gender_count = df.groupby("Sex")["HeartDisease"].value_counts().unstack()
print(gender_count)
print(type(gender_count))
print(gender_count.index)
print(gender_count.values)

# Does age influence heart disease?

'AGE distribution'
plt.hist(df['Age'],bins=15,color="skyblue",edgecolor="black")
plt.title("Age Distribution of Patients", fontsize=15)
plt.xlabel("Age",fontsize=15,color="blue")
plt.ylabel("Number of Patients",fontsize=15,color="blue")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

'now patients with and without heart disease'
plt.figure(figsize=(12,5))
heart_yes = df[df["HeartDisease"] == 1]["Age"]
heart_no = df[df["HeartDisease"] == 0]["Age"]

plt.subplot(1,2,1)
plt.hist(heart_yes,bins=15,alpha=0.6,color="red",label="Heart Disease",edgecolor="black")
plt.title("Age Distribution Patients WITH Heart Disease",fontsize=12,color="olive")
plt.xlabel("Age",fontsize=12,color="olive")
plt.ylabel("Patients with heart disease",fontsize=12,color="olive")
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1,2,2)
plt.hist(heart_no,bins=15,alpha=0.6,color="skyblue",label="No Heart Disease",edgecolor="black")
plt.title("Age Distribution Patients WITHOUT Heart Disease",fontsize=12,color="olive")
plt.xlabel("Age",fontsize=12,color="skyblue")
plt.ylabel("Patients with no heart disease",fontsize=12,color="skyblue")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Does the type of chest pain influence the likelihood of heart disease?

chest_pain_dist=df.groupby("ChestPainType")["HeartDisease"].value_counts()
print(chest_pain_dist)

plt.figure(figsize=(8,5))
sns.countplot(data=df,x="ChestPainType",hue="HeartDisease",palette=["steelblue", "crimson"])

plt.title("Chest Pain Type vs Heart Disease",fontsize=16,color="darkblue",fontweight="bold")
plt.xlabel("Chest Pain Type",fontsize=13,color="darkgreen")
plt.ylabel("Number of Patients",fontsize=13,color="darkgreen")
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.legend(title="Heart Disease",fontsize=11,title_fontsize=12)
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.show()

# Does cholesterol have a relationship with heart disease?
avg_chol=df.groupby("HeartDisease")["Cholesterol"].mean()
print(avg_chol)
median_chol=df.groupby("HeartDisease")["Cholesterol"].median()
print(median_chol)

plt.figure(figsize=(8,5))

plt.hist(df["Cholesterol"],bins=20,color="skyblue",edgecolor="black")

plt.title("Distribution of Cholesterol",fontsize=16,fontweight="bold")
plt.xlabel("Cholesterol", fontsize=13)
plt.ylabel("Number of Patients", fontsize=13)

plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print("patients having 0 cholesterol: ",(df["Cholesterol"]==0).sum()) # 172 people have 0 cholesterol which is not possible

df["Cholesterol"]=df["Cholesterol"].replace(0,np.nan)
df["Cholesterol"]=df["Cholesterol"].fillna(df["Cholesterol"].median())
#Because cholesterol can have high outliers , and the median is less affected by extreme values.


# Do patients with heart disease have higher resting blood pressure?
avg_restbp=df.groupby("HeartDisease")["RestingBP"].describe()
print(avg_restbp)

plt.figure(figsize=(12,5))

heart_yes = df[df["HeartDisease"] == 1]["RestingBP"]
heart_no = df[df["HeartDisease"] == 0]["RestingBP"]

# ---------------- Left ----------------
plt.subplot(1,2,1)

plt.hist(heart_yes,bins=20,color="crimson",alpha=0.7,edgecolor="black",label="Heart Disease")

plt.title("Resting BP - Patients WITH Heart Disease",fontsize=13,fontweight="bold",color="darkred")

plt.xlabel("Resting Blood Pressure (mmHg)",fontsize=12)
plt.ylabel("Number of Patients",fontsize=12)

plt.legend()
plt.grid(alpha=0.3)

# ---------------- Right ----------------
plt.subplot(1,2,2)

plt.hist(heart_no,bins=20,color="steelblue",alpha=0.7,edgecolor="black",label="No Heart Disease")

plt.title("Resting BP - Patients WITHOUT Heart Disease",fontsize=13,fontweight="bold",color="navy")

plt.xlabel("Resting Blood Pressure (mmHg)",fontsize=12)
plt.ylabel("Number of Patients",fontsize=12)

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.show()

# Does maximum heart rate differ between patients with and without heart disease?
avg_heart_rate=df.groupby("HeartDisease")["MaxHR"].describe()
print(avg_heart_rate)

plt.figure(figsize=(12,5))

heart_yes = df[df["HeartDisease"] == 1]["MaxHR"]
heart_no = df[df["HeartDisease"] == 0]["MaxHR"]

# ---------------- Left ----------------
plt.subplot(1,2,1)

plt.hist(heart_yes,bins=20,color="crimson",alpha=0.7,edgecolor="black",label="Heart Disease")

plt.title("Maximum Heart Rate - Patients WITH Heart Disease",fontsize=13,fontweight="bold",color="darkred")

plt.xlabel("Maximum Heart Rate",fontsize=12)
plt.ylabel("Number of Patients",fontsize=12)

plt.legend()
plt.grid(alpha=0.3)

# ---------------- Right ----------------
plt.subplot(1,2,2)

plt.hist(heart_no,bins=20,color="steelblue",alpha=0.7,edgecolor="black",label="No Heart Disease")

plt.title("Maximum Heart Rate - Patients WITHOUT Heart Disease",fontsize=13,fontweight="bold",color="navy")

plt.xlabel("Maximum Heart Rate",fontsize=12)
plt.ylabel("Number of Patients",fontsize=12)

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.show()

# resting ecg vs heart disease
plt.figure(figsize=(8,5))

sns.countplot(data=df,x="RestingECG",hue="HeartDisease",palette=["steelblue","crimson"])

plt.title("Resting ECG vs Heart Disease", fontsize=16, fontweight="bold")
plt.xlabel("Resting ECG", fontsize=12)
plt.ylabel("Number of Patients", fontsize=12)

plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.show()

# ST slope vs heart disease
plt.figure(figsize=(8,5))

sns.countplot(data=df,x="ST_Slope",hue="HeartDisease",palette=["steelblue","crimson"])

plt.title("ST Slope vs Heart Disease", fontsize=16, fontweight="bold")
plt.xlabel("ST Slope", fontsize=12)
plt.ylabel("Number of Patients", fontsize=12)

plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.show()