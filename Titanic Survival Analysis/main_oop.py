import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from time import sleep as delay
from random import randint as rand

class TitanicDataAnalysis:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None

    # 1. Generate Data
    def generate_data(self):
        if os.path.exists("data_generate.py") and not os.path.exists(
            "titanic_survival_dataset.csv"
        ):
            os.system("python data_generate.py")
            print("✅ Data generated successfully.")
        elif not os.path.exists("titanic_survival_dataset.csv"):
            print("❌ data already exists!")
        elif not os.path.exists("data_generate.py"):
            print("❌ data_generate.py not found!")

    # 2. Load Data
    def load_data(self):
        if not os.path.exists(self.file_path):
            print("⚠️ Data file not found!")
            return
        self.df = pd.read_csv(self.file_path)
        print("✅ Data loaded successfully.")
        print("Shape:", self.df.shape)

    # 3. Basic Info
    def basic_info(self):
        if self.df is None:
            print("⚠️ Data not loaded.")
            return
        print("\n--- Basic Info ---")
        print("Dataset Shape:", self.df.shape)
        print("\nColumn Data Types:\n", self.df.dtypes)
        print("\nMissing Values:\n", self.df.isna().sum())
        print("\nMissing Values (%):\n", (self.df.isna().sum() / len(self.df)) * 100)

    # 4. Handle Missing Values
    def handle_missing_values(self):
        if self.df is None:
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
                                self.df[col] = self.df[col].fillna(self.df[col].mean())
                            else:
                                self.df[col] = self.df[col].fillna(self.df[col].median())
                        print(f"✅ Missing values filled using {func}.")
                    else:
                        print("❌ Invalid function!")
                case 2:
                    self.df.dropna(inplace=True)
                    print("✅ Rows with missing values dropped.")
                case 0:
                    print("Exiting missing value handler...")
                    break
                case _:
                    print("❌ Invalid choice!")

    # 5. All Analysis
    def all_analysis(self):
        if self.df is None:
            print("⚠️ Data not loaded.")
            return
        df = self.df
        # ===============================
        # 1. Basic Info
        # ===============================
        print("Dataset Shape:", df.shape)
        print("\nColumn Data Types:\n", df.dtypes)
        print("\nMissing Values:\n", df.isna().sum())

        numeric_cols = ["Age", "Fare", "SibSp", "Parch", "Pclass"]
        print("\nSummary Statistics:\n", df[numeric_cols].describe(include="all"))

        missing_percent = (df.isna().sum() / len(df)) * 100
        print("\nMissing Values (%):\n", missing_percent)

        # ===============================
        # 2. Survival Analysis
        # ===============================
        survival_rate = df["Survived"].mean()
        print("\nOverall Survival Rate:", survival_rate)

        survival_by_class = df.groupby("Pclass")["Survived"].mean().sort_values(ascending=False)
        print("\nSurvival Rate by Passenger Class:\n", survival_by_class)

        survival_by_sex = df.groupby("Sex")["Survived"].mean().sort_values(ascending=False)
        print("\nSurvival Rate by Sex:\n", survival_by_sex)

        survival_by_embarked = df.groupby("Embarked")["Survived"].mean().sort_values(ascending=False)
        print("\nSurvival Rate by Embarked Port:\n", survival_by_embarked)

        # ===============================
        # 3. Age & Fare Analysis
        # ===============================
        avg_age_survival = df.groupby("Survived")["Age"].mean()
        print("\nAverage Age by Survival:\n", avg_age_survival)

        avg_fare_survival = df.groupby("Survived")["Fare"].mean()
        print("\nAverage Fare by Survival:\n", avg_fare_survival)

        # ===============================
        # 4. Correlations
        # ===============================
        corr_matrix = df[numeric_cols + ["Survived"]].corr()
        print("\nCorrelation Matrix:\n", corr_matrix)

        # ===============================
        # 5. Family Analysis (SibSp + Parch)
        # ===============================
        df["Family_Size"] = df["SibSp"] + df["Parch"] + 1
        survival_by_family = df.groupby("Family_Size")["Survived"].mean()
        print("\nSurvival Rate by Family Size:\n", survival_by_family.head(10))

        # ===============================
        # 6. Derived Metrics
        # ===============================
        # Example: Age to Fare ratio
        df["Age_to_Fare"] = df["Age"] / df["Fare"].replace(0, np.nan)
        print("\nGlobal Average Age-to-Fare Ratio:", df["Age_to_Fare"].mean())

        # Example: Child indicator (Age < 12)
        df["Child"] = df["Age"].apply(lambda x: 1 if x < 12 else 0)
        child_survival = df.groupby("Child")["Survived"].mean()
        print("\nSurvival Rate for Children vs Adults:\n", child_survival)

        # Example: Alone indicator
        df["Alone"] = (df["SibSp"] + df["Parch"] == 0).astype(int)
        alone_survival = df.groupby("Alone")["Survived"].mean()
        print("\nSurvival Rate for Alone vs With Family:\n", alone_survival)

    # 6. All Visualizations
    def all_visualizations(self):
        if self.df is None:
            print("⚠️ Data not loaded.")
            return
        df = self.df
        numeric_cols = ["Age", "Fare", "SibSp", "Parch", "Pclass"]

        # ===============================
        # 1. Univariate Analysis
        # ===============================

        # Histograms
        for col in numeric_cols:
            plt.figure(figsize=(6, 4))
            sns.histplot(df[col].dropna(), bins=30, kde=True)
            plt.title(f"Distribution of {col}")
            plt.show()

        # Boxplots
        for col in numeric_cols:
            plt.figure(figsize=(6, 4))
            sns.boxplot(x=df[col])
            plt.title(f"Boxplot of {col}")
            plt.show()

        # ===============================
        # 2. Bivariate Analysis
        # ===============================

        # Scatterplots (numeric vs Survived)
        pairs = [
            ("Age", "Survived"),
            ("Fare", "Survived"),
            ("SibSp", "Survived"),
            ("Parch", "Survived"),
            ("Pclass", "Survived"),
        ]

        for x, y in pairs:
            plt.figure(figsize=(6, 4))
            sns.scatterplot(data=df, x=x, y=y, alpha=0.5)
            plt.title(f"{x} vs {y}")
            plt.show()

        # Correlation heatmap
        plt.figure(figsize=(10, 6))
        sns.heatmap(
            df[numeric_cols + ["Survived"]].corr(), annot=True, cmap="coolwarm", fmt=".2f"
        )
        plt.title("Correlation Heatmap")
        plt.show()

        # Pairplot (sample for speed)
        sns.pairplot(
            df[numeric_cols + ["Survived"]].dropna().sample(500, random_state=42),
            hue="Survived",
        )
        plt.show()

        # ===============================
        # 3. Categorical Analysis
        # ===============================

        # Survival by Sex
        plt.figure(figsize=(6, 4))
        sns.countplot(data=df, x="Sex", hue="Survived")
        plt.title("Survival Count by Sex")
        plt.show()

        # Survival by Embarked
        plt.figure(figsize=(6, 4))
        sns.countplot(data=df, x="Embarked", hue="Survived")
        plt.title("Survival Count by Embarked Port")
        plt.show()

        # Survival by Pclass
        plt.figure(figsize=(6, 4))
        sns.countplot(data=df, x="Pclass", hue="Survived")
        plt.title("Survival Count by Passenger Class")
        plt.show()

        # ===============================
        # 4. Advanced Visualizations
        # ===============================

        # Fare distribution by survival
        plt.figure(figsize=(8, 6))
        sns.violinplot(data=df, x="Survived", y="Fare")
        plt.title("Fare Distribution by Survival")
        plt.show()

        # Age distribution by survival
        plt.figure(figsize=(8, 6))
        sns.violinplot(data=df, x="Survived", y="Age")
        plt.title("Age Distribution by Survival")
        plt.show()

        # Heatmap of survival rate by Sex & Pclass
        pivot = df.pivot_table(values="Survived", index="Sex", columns="Pclass", aggfunc="mean")
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            pivot, annot=True, cmap="YlGnBu", fmt=".2f", cbar_kws={"label": "Survival Rate"}
        )
        plt.title("Survival Rate by Sex & Pclass")
        plt.show()

        # FacetGrid: Age distribution by survival & sex
        g = sns.FacetGrid(df, col="Survived", row="Sex", height=3.5, margin_titles=True)
        g.map_dataframe(sns.histplot, x="Age", bins=20, kde=True)
        g.set_axis_labels("Age", "Count")
        plt.show()


# ==========================
# 🚀 Menu-driven interaction
# ==========================
def main():
    file_path = "df.csv"
    analyzer = TitanicDataAnalysis(file_path)

    menu = {
        1: ("Create Dataset", analyzer.generate_data),
        2: ("Load Data", analyzer.load_data),
        3: ("Basic Info", analyzer.basic_info),
        4: ("Handle Missing Values", analyzer.handle_missing_values),
        5: ("All Analysis", analyzer.all_analysis),
        6: ("All Visualizations", analyzer.all_visualizations),
        0: ("Exit", None),
    }

    while True:
        print("\n--- Titanic Data Analysis Menu ---")
        for k,v in menu.items():
            print(f"{k}. {v[0]}")
        choice = int(input("Enter your choice: "))

        if choice == 0:
            print("Exiting...")
            delay(rand(1,3))
            break
        elif choice in menu:
            menu[choice][1]()
        else:
            print("❌ Invalid choice!")

if __name__ == "__main__":
    main()
