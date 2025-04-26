# Import necessary library
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats
import squarify  # For Treemap


# Step 1: Load the dataset
df = pd.read_csv('employees_project.csv')  # Adjust the path if needed
print("✅ Data Loaded Successfully!\n")

# Step 2: Check basic info
print("🔎 Dataset Info:")
print(df.info())
print("\n")

# Step 3: Summary statistics
print("📊 Summary Statistics (Numerical):")
print(df.describe())
print("\n")

print("📋 Summary Statistics (Categorical):")
print(df.describe(include=['object']))
print("\n")

# Step 4: Check for missing values
print("🚨 Missing Values per Column:")
print(df.isnull().sum())
print("\n")

# Step 5: Check for duplicate rows
duplicates = df.duplicated().sum()
print(f"🔁 Number of Duplicate Rows: {duplicates}\n")

# Remove duplicates if found
if duplicates > 0:
    df = df.drop_duplicates()
    print("✅ Duplicates removed.\n")

# Step 6: Convert columns to appropriate data types
categorical_cols = ['Department', 'Gender', 'JobLevel']
for col in categorical_cols:
    df[col] = df[col].astype('category')

print("✅ Data types corrected for categorical columns.\n")

# Step 7: Final check
print("✅ Final Dataset Info:")
print(df.info())

# Step 8: Save the cleaned data (optional)
df.to_csv('employees_project_cleaned.csv', index=False)
print("\n📂 Cleaned file saved as 'employees_project_cleaned.csv'")

# # # ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## # #
# Data Exploration Analysis EDA

# Step 1: Load the cleaned data
df = pd.read_csv('employees_project_cleaned.csv')
print("✅ Cleaned Data Loaded!\n")

# Step 2: Overview of the dataset
print("🔎 Dataset Shape (Rows, Columns):", df.shape)
print("\n")

print("📋 Columns List:")
print(df.columns.tolist())
print("\n")

# Step 3: Unique values in categorical columns
categorical_cols = ['Department', 'Gender', 'JobLevel']

for col in categorical_cols:
    print(f"🧹 Unique values in {col}: {df[col].unique()}")
    print("\n")

# Step 4: Distribution of Numerical Features
print("📊 Distribution of Numerical Columns:\n")
print(df.describe())

# Step 5: Count of Employees in each Department
print("🏢 Employee Count per Department:")
print(df['Department'].value_counts())
print("\n")

# Step 6: Count of Employees by Gender
print("👨‍🦰👩 Gender Distribution:")
print(df['Gender'].value_counts())
print("\n")

# Step 7: Correlation Matrix
print("🔗 Correlation between Numerical Columns:\n")
print(df.corr())
print("\n")

# Step 8: Quick Visualizations

# Bar Plot: Employee count per department
plt.figure(figsize=(8,5))
sns.countplot(data=df, x='Department', palette='viridis')
plt.title('Employee Count per Department')
plt.xticks(rotation=45)
plt.show()

# Histogram: Salary Distribution
plt.figure(figsize=(8,5))
sns.histplot(df['Salary'], kde=True, color='blue')
plt.title('Salary Distribution')
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# # # # # # # # # ## ## ## ## ## ## ## ## ## ## ## # # #
# Full Statistical Analysis

# Step 1: Load the cleaned data
df = pd.read_csv('employees_project_cleaned.csv')
print("✅ Cleaned Data Loaded for Statistics!\n")

# Step 2: Normal Distribution Analysis (e.g., Salary)
salary_mean = df['Salary'].mean()
salary_std = df['Salary'].std()

print(f"📊 Salary Mean: {salary_mean}")
print(f"📊 Salary Standard Deviation: {salary_std}\n")

# Plot normal distribution for Salary
sns.histplot(df['Salary'], kde=True, color='skyblue')
plt.title('Salary Normal Distribution')
plt.xlabel('Salary')
plt.ylabel('Frequency')
plt.show()

# Step 3: Confidence Interval for Salary (95%)
confidence = 0.95
n = len(df['Salary'])
m = salary_mean
se = stats.sem(df['Salary'])  # Standard error
h = se * stats.t.ppf((1 + confidence) / 2, n-1)

print(f"🎯 95% Confidence Interval for Salary: ({m-h:.2f}, {m+h:.2f})\n")

# Step 4: Binomial Distribution Example (e.g., Gender Split: Male or not Male)
# Encoding 'Male' as 1, others as 0
df['Male_Flag'] = np.where(df['Gender'] == 'Male', 1, 0)

n_male = df['Male_Flag'].sum()
n_total = df['Male_Flag'].count()

p_male = n_male / n_total

print(f"👨 Probability of Male Employee: {p_male:.2f}\n")

# Step 5: Poisson Distribution Example (Events like "High Performance Scores")
# Define high performance as PerformanceScore > 8
high_perf = (df['PerformanceScore'] > 8).sum()

# Poisson expects "number of events" and "time interval",
# here we just simulate assuming one time period (1 year)
lambda_val = high_perf / 1

print(f"💥 Estimated lambda for High Performers: {lambda_val:.2f}\n")

# Poisson PMF (Probability of getting exactly k high performers)
k = 5  # Example: probability of exactly 5 high performers
poisson_prob = stats.poisson.pmf(k, lambda_val)
print(f"🔢 Probability of exactly {k} high performers: {poisson_prob:.4f}")

# # # # # # # # # # # ## ## ## ## ## ## ## ## # # # # # #
# Full Data Visualization
# Load the cleaned data
df = pd.read_csv('employees_project_cleaned.csv')
print("✅ Data loaded for visualization!\n")

# Set a general style
sns.set(style="whitegrid")

# 1. Bar Chart — Employees per Department
plt.figure(figsize=(8,5))
sns.countplot(data=df, x='Department', palette='Set2')
plt.title('Number of Employees in Each Department')
plt.xticks(rotation=45)
plt.show()

# 2. Line Chart — Salary vs YearsExperience
plt.figure(figsize=(8,5))
sns.lineplot(data=df, x='YearsExperience', y='Salary', marker='o', color='coral')
plt.title('Salary Growth with Experience')
plt.show()

# 3. Histogram — Bonus % distribution
plt.figure(figsize=(8,5))
sns.histplot(df['Bonus %'], bins=20, kde=True, color='teal')
plt.title('Bonus % Distribution')
plt.show()

# 4. Boxplot — Salary by Gender
plt.figure(figsize=(8,5))
sns.boxplot(data=df, x='Gender', y='Salary', palette='pastel')
plt.title('Salary Distribution by Gender')
plt.show()

# 5. Scatter Plot — Salary vs PerformanceScore
plt.figure(figsize=(8,5))
sns.scatterplot(data=df, x='PerformanceScore', y='Salary', hue='Gender')
plt.title('Salary vs Performance Score')
plt.show()

# 6. Heatmap — Correlation between Numerical Columns
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap='YlGnBu')
plt.title('Correlation Matrix')
plt.show()

# 7. Pie Chart — Gender Distribution
gender_counts = df['Gender'].value_counts()
plt.figure(figsize=(6,6))
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=140, colors=['lightblue', 'pink', 'lightgreen'])
plt.title('Gender Distribution')
plt.show()

# 8. Treemap — Department-wise employee distribution
dept_counts = df['Department'].value_counts()
plt.figure(figsize=(12,8))
squarify.plot(sizes=dept_counts.values, label=dept_counts.index, alpha=0.8)
plt.title('Employee Distribution by Department (Treemap)')
plt.axis('off')
plt.show()
