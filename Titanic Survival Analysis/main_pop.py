import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from time import sleep as delay
from random import randint as rand

# Global dataframe
df = None
file_path = "titanic_survival_dataset.csv"

# ===============================
# 1. Generate Data
# ===============================
def generate_data():
    if os.path.exists("data_generate.py") and not os.path.exists("titanic_survival_dataset.csv"):
        os.system("python data_generate.py")
        print("✅ Data generated successfully.")
    elif os.path.exists("titanic_survival_dataset.csv"):
        print("⚠️ Data already exists!")
    else:
        print("❌ data_generate.py not found!")

# ===============================
# 2. Load Data
# ===============================
def load_data():
    global df
    if not os.path.exists(file_path):
        print("⚠️ Data file not found!")
        return
    df = pd.read_csv(file_path)
    print("✅ Data loaded successfully.")
    print("Shape:", df.shape)

# ===============================
# 3. Basic Info
# ===============================
def basic_info():
    global df
    if df is None:
        print("⚠️ Data not loaded.")
        return
    print("\n--- Basic Info ---")
    print("Dataset Shape:", df.shape)
    print("\nColumn Data Types:\n", df.dtypes)
    print("\nMissing Values:\n", df.isna().sum())
    print("\nMissing Values (%):\n", (df.isna().sum() / len(df)) * 100)

# ===============================
# 4. Handle Missing Values
# ===============================
def handle_missing_values():
    global df
    if df is None:
        print("⚠️ Data not loaded.")
        return
    numeric_cols = ["Age", "Fare", "SibSp", "Parch", "Pclass"]
    while True:
        print("\n--- Handling Missing Values ---")
        print("1. Fill numeric columns with mean/median")
        print("2. Drop rows with missing values")
        print("0. Exit")
        choice = int(input("Enter your choice: "))

        match choice:
            case 1:
                func = input("Enter aggregate function (mean/median): ").strip().lower()
                if func in ["mean", "median"]:
                    for col in numeric_cols:
                        if func == "mean":
                            df[col] = df[col].fillna(df[col].mean())
                        else:
                            df[col] = df[col].fillna(df[col].median())
                    print(f"✅ Missing values filled using {func}.")
                else:
                    print("❌ Invalid function!")
            case 2:
                df.dropna(inplace=True)
                print("✅ Rows with missing values dropped.")
            case 0:
                print("Exiting missing value handler...")
                break
            case _:
                print("❌ Invalid choice!")

# ===============================
# 5. All Analysis
# ===============================
def all_analysis():
    global df
    if df is None:
        print("⚠️ Data not loaded.")
        return
    numeric_cols = ["Age", "Fare", "SibSp", "Parch", "Pclass"]

    # 1. Basic Info
    print("Dataset Shape:", df.shape)
    print("\nColumn Data Types:\n", df.dtypes)
    print("\nMissing Values:\n", df.isna().sum())
    print("\nSummary Statistics:\n", df[numeric_cols].describe(include="all"))
    print("\nMissing Values (%):\n", (df.isna().sum() / len(df)) * 100)

    # 2. Survival Analysis
    print("\nOverall Survival Rate:", df["Survived"].mean())
    print("\nSurvival Rate by Passenger Class:\n", df.groupby("Pclass")["Survived"].mean())
    print("\nSurvival Rate by Sex:\n", df.groupby("Sex")["Survived"].mean())
    print("\nSurvival Rate by Embarked Port:\n", df.groupby("Embarked")["Survived"].mean())

    # 3. Age & Fare Analysis
    print("\nAverage Age by Survival:\n", df.groupby("Survived")["Age"].mean())
    print("\nAverage Fare by Survival:\n", df.groupby("Survived")["Fare"].mean())

    # 4. Correlations
    print("\nCorrelation Matrix:\n", df[numeric_cols + ["Survived"]].corr())

    # 5. Family Analysis
    df["Family_Size"] = df["SibSp"] + df["Parch"] + 1
    print("\nSurvival Rate by Family Size:\n", df.groupby("Family_Size")["Survived"].mean().head(10))

    # 6. Derived Metrics
    df["Age_to_Fare"] = df["Age"] / df["Fare"].replace(0, np.nan)
    print("\nGlobal Average Age-to-Fare Ratio:", df["Age_to_Fare"].mean())

    df["Child"] = df["Age"].apply(lambda x: 1 if x < 12 else 0)
    print("\nSurvival Rate for Children vs Adults:\n", df.groupby("Child")["Survived"].mean())

    df["Alone"] = (df["SibSp"] + df["Parch"] == 0).astype(int)
    print("\nSurvival Rate for Alone vs With Family:\n", df.groupby("Alone")["Survived"].mean())

# ===============================
# 6. All Visualizations
# ===============================
def all_visualizations():
    global df
    if df is None:
        print("⚠️ Data not loaded.")
        return
    numeric_cols = ["Age", "Fare", "SibSp", "Parch", "Pclass"]

    # Univariate Analysis
    for col in numeric_cols:
        plt.figure(figsize=(6,4))
        sns.histplot(df[col].dropna(), bins=30, kde=True)
        plt.title(f"Distribution of {col}")
        plt.show()

    for col in numeric_cols:
        plt.figure(figsize=(6,4))
        sns.boxplot(x=df[col])
        plt.title(f"Boxplot of {col}")
        plt.show()

    # Bivariate Analysis
    pairs = [("Age","Survived"),("Fare","Survived"),("SibSp","Survived"),("Parch","Survived"),("Pclass","Survived")]
    for x,y in pairs:
        plt.figure(figsize=(6,4))
        sns.scatterplot(data=df, x=x, y=y, alpha=0.5)
        plt.title(f"{x} vs {y}")
        plt.show()

    plt.figure(figsize=(10,6))
    sns.heatmap(df[numeric_cols + ["Survived"]].corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.show()

    sns.pairplot(df[numeric_cols + ["Survived"]].dropna().sample(500, random_state=42), hue="Survived")
    plt.show()

    # Categorical Analysis
    plt.figure(figsize=(6,4))
    sns.countplot(data=df, x="Sex", hue="Survived")
    plt.title("Survival Count by Sex")
    plt.show()

    plt.figure(figsize=(6,4))
    sns.countplot(data=df, x="Embarked", hue="Survived")
    plt.title("Survival Count by Embarked Port")
    plt.show()

    plt.figure(figsize=(6,4))
    sns.countplot(data=df, x="Pclass", hue="Survived")
    plt.title("Survival Count by Passenger Class")
    plt.show()

    # Advanced Visualizations
    plt.figure(figsize=(8,6))
    sns.violinplot(data=df, x="Survived", y="Fare")
    plt.title("Fare Distribution by Survival")
    plt.show()

    plt.figure(figsize=(8,6))
    sns.violinplot(data=df, x="Survived", y="Age")
    plt.title("Age Distribution by Survival")
    plt.show()

    pivot = df.pivot_table(values="Survived", index="Sex", columns="Pclass", aggfunc="mean")
    plt.figure(figsize=(8,6))
    sns.heatmap(pivot, annot=True, cmap="YlGnBu", fmt=".2f", cbar_kws={'label': 'Survival Rate'})
    plt.title("Survival Rate by Sex & Pclass")
    plt.show()

    g = sns.FacetGrid(df, col="Survived", row="Sex", height=3.5, margin_titles=True)
    g.map_dataframe(sns.histplot, x="Age", bins=20, kde=True)
    g.set_axis_labels("Age", "Count")
    plt.show()

# ==========================
# 🚀 Menu-driven interaction
# ==========================
def main():
    menu = {
        1: ("Create Dataset", generate_data),
        2: ("Load Data", load_data),
        3: ("Basic Info", basic_info),
        4: ("Handle Missing Values", handle_missing_values),
        5: ("All Analysis", all_analysis),
        6: ("All Visualizations", all_visualizations),
        0: ("Exit", None),
    }

    while True:
        print("\n--- Titanic Data Analysis Menu ---")
        for k,v in menu.items():
            print(f"{k}. {v[0]}")
        choice = int(input("Enter your choice: "))

        match choice:
            case 0:
                print("Exiting...")
                delay(rand(1,3))
                break
            case _ if choice in menu:
                menu[choice][1]()
            case _:
                print("❌ Invalid choice!")

if __name__ == "__main__":
    main()
