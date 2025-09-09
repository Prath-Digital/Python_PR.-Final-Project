import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from time import sleep as delay
from random import randint as rand
import os


class HappinessDataAnalysis:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None

    def generate_data(self):
        if os.path.exists("data_generate.py") and not os.path.exists(
            "global_happiness_report.csv"
        ):
            os.system("python data_generate.py")
            print("✅ Data generated successfully.")
        elif not os.path.exists("global_happiness_report.csv"):
            print("❌ data already exists!")
        elif not os.path.exists("data_generate.py"):
            print("❌ data_generate.py not found!")

    # 1. Load Data
    def load_data(self):
        if not os.path.exists(self.file_path):
            print("⚠️ Data file not found!")
            return

        self.df = pd.read_csv(self.file_path)
        self.df["Date"] = pd.to_datetime(self.df["Date"], errors="coerce")
        print("✅ Data loaded successfully.")
        print("Shape:", self.df.shape)

    # 2. Basic Info
    def basic_info(self):
        if self.df is None:
            print("⚠️ Data not loaded.")
            return
        print("\n--- Basic Info ---")
        print(self.df.info())
        print("\nMissing Values:\n", self.df.isna().sum())
        print("\nMissing Values (%):\n", (self.df.isna().sum() / len(self.df)) * 100)

    # 3. Handle Missing Values
    def handle_missing_values(self):
        if self.df is None:
            print("⚠️ Data not loaded.")
            return
        numeric_cols = [
            "Happiness_Score",
            "GDP_Per_Capita",
            "Social_Support",
            "Healthy_Life_Expectancy",
            "Freedom_To_Make_Life_Choices",
            "Generosity",
            "Perceptions_Of_Corruption",
            "Positive_Affect",
            "Negative_Affect",
            "Confidence_In_Government",
        ]
        while True:
            print("\n--- Handling Missing Values ---")
            print("1. Fill numeric columns with mean/median")
            print("2. Drop rows with missing values")
            print("0. Exit")
            choice = int(input("Enter your choice: "))

            match choice:
                case 1:
                    func = (
                        input("Enter aggregate function (mean/median): ")
                        .strip()
                        .lower()
                    )
                    if func in ["mean", "median"]:
                        for col in numeric_cols:
                            if func == "mean":
                                self.df[col] = self.df[col].fillna(self.df[col].mean())
                            else:
                                self.df[col] = self.df[col].fillna(
                                    self.df[col].median()
                                )
                        print(f"✅ Missing values filled using {func}.")
                    else:
                        print("❌ Invalid function!")
                case 2:
                    self.df.dropna(inplace=True)
                    print("✅ Rows with missing values dropped.")
                case 0:
                    print("No changes made.")
                    print("Exiting...")
                    break
                case _:
                    print("❌ Invalid choice!")

    # 4. All Analysis
    def all_analysis(self):
        # ===============================
        # 1. Basic Info
        # ===============================
        if self.df is None:
            print("⚠️ Data not loaded. Please load data first.")
            return
        df = self.df
        print("Dataset Shape:", df.shape)
        print("\nColumn Data Types:\n", df.dtypes)
        print("\nMissing Values:\n", df.isna().sum())

        numeric_cols = [
            "Happiness_Score",
            "GDP_Per_Capita",
            "Social_Support",
            "Healthy_Life_Expectancy",
            "Freedom_To_Make_Life_Choices",
            "Generosity",
            "Perceptions_Of_Corruption",
            "Positive_Affect",
            "Negative_Affect",
            "Confidence_In_Government",
        ]
        print("\nSummary Statistics:\n", df[numeric_cols].describe(include="all"))

        missing_percent = (df.isna().sum() / len(df)) * 100
        print("\nMissing Values (%):\n", missing_percent)

        # ===============================
        # 2. Country-level Analysis
        # ===============================
        avg_happiness = (
            df.groupby("Country")["Happiness_Score"].mean().sort_values(ascending=False)
        )
        print(
            "\nTop 10 Countries by Average Happiness Score:\n", avg_happiness.head(10)
        )

        avg_gdp = (
            df.groupby("Country")["GDP_Per_Capita"].mean().sort_values(ascending=False)
        )
        print("\nTop 10 Countries by Avg GDP Per Capita:\n", avg_gdp.head(10))

        avg_social_support = (
            df.groupby("Country")["Social_Support"].mean().sort_values(ascending=False)
        )
        print(
            "\nTop 10 Countries by Avg Social Support:\n", avg_social_support.head(10)
        )

        avg_life_expectancy = (
            df.groupby("Country")["Healthy_Life_Expectancy"]
            .mean()
            .sort_values(ascending=False)
        )
        print(
            "\nTop 10 Countries by Healthy Life Expectancy:\n",
            avg_life_expectancy.head(10),
        )

        # ===============================
        # 3. Time Series Analysis
        # ===============================
        time_series = df.groupby("Date")[
            ["Happiness_Score", "Positive_Affect", "Negative_Affect"]
        ].mean()
        print("\nOverall Time Series (first 10 rows):\n", time_series.head(10))

        # Daily change in Happiness Score
        time_series["Daily_Happiness_Change"] = time_series["Happiness_Score"].diff()
        print(
            "\nDaily Change in Happiness Score (first 10 rows):\n",
            time_series["Daily_Happiness_Change"].head(10),
        )

        # ===============================
        # 4. Correlations
        # ===============================
        corr_matrix = df[numeric_cols].corr()
        print("\nCorrelation Matrix:\n", corr_matrix)

        # ===============================
        # 5. Region-level Analysis
        # ===============================
        state_happiness = (
            df.groupby(["Country", "State_Region"])["Happiness_Score"]
            .mean()
            .sort_values(ascending=False)
        )
        print(
            "\nTop 10 States/Regions by Average Happiness Score:\n",
            state_happiness.head(10),
        )

        # ===============================
        # 6. Derived Metrics
        # ===============================
        # Example: Ratio of Positive to Negative Affect
        df["Affect_Ratio"] = df["Positive_Affect"] / df["Negative_Affect"].replace(
            0, np.nan
        )

        print(
            "\nGlobal Average Positive/Negative Affect Ratio:",
            df["Affect_Ratio"].mean(),
        )

        # Example: Happiness to GDP ratio
        df["Happiness_to_GDP"] = df["Happiness_Score"] / df["GDP_Per_Capita"].replace(
            0, np.nan
        )
        print("Global Average Happiness-to-GDP Ratio:", df["Happiness_to_GDP"].mean())

    # 5. All Visualizations
    def all_visualizations(self):
        if self.df is None:
            print("⚠️ Data not loaded. Please load data first.")
            return
        df = self.df
        numeric_cols = [
            "Happiness_Score",
            "GDP_Per_Capita",
            "Social_Support",
            "Healthy_Life_Expectancy",
            "Freedom_To_Make_Life_Choices",
            "Generosity",
            "Perceptions_Of_Corruption",
            "Positive_Affect",
            "Negative_Affect",
            "Confidence_In_Government",
        ]
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
        # Scatterplots
        pairs = [
            ("GDP_Per_Capita", "Happiness_Score"),
            ("Social_Support", "Happiness_Score"),
            ("Healthy_Life_Expectancy", "Happiness_Score"),
            ("Freedom_To_Make_Life_Choices", "Happiness_Score"),
        ]

        for x, y in pairs:
            plt.figure(figsize=(6, 4))
            sns.scatterplot(data=df, x=x, y=y, alpha=0.5)
            plt.title(f"{x} vs {y}")
            plt.show()

        # Correlation heatmap
        plt.figure(figsize=(10, 6))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Heatmap")
        plt.show()

        # Pairplot (sample for speed)
        sns.pairplot(df[numeric_cols].dropna().sample(500, random_state=42))
        plt.show()

        # ===============================
        # 3. Time Series Analysis
        # ===============================
        if "Date" in df.columns:
            time_group = df.groupby("Date")[
                ["Happiness_Score", "Positive_Affect", "Negative_Affect"]
            ].mean()

            # Trends over time
            plt.figure(figsize=(12, 6))
            time_group.plot()
            plt.title("Happiness, Positive & Negative Affect Over Time")
            plt.ylabel("Scores")
            plt.show()

            # Rolling average (30-day)
            time_group_rolling = time_group.rolling(30).mean()
            time_group_rolling.plot(figsize=(12, 6))
            plt.title("30-Day Rolling Average")
            plt.ylabel("Scores")
            plt.show()

        # ===============================
        # 4. Country & Region Analysis
        # ===============================
        # Top 10 countries by Happiness Score
        top_countries = df.groupby("Country")["Happiness_Score"].mean().nlargest(10)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=top_countries.values, y=top_countries.index)
        plt.title("Top 10 Countries by Average Happiness Score")
        plt.show()

        # Boxplot of GDP by Country (top 5)
        top5 = df["Country"].value_counts().head(5).index
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df[df["Country"].isin(top5)], x="Country", y="GDP_Per_Capita")
        plt.title("GDP per Capita Distribution (Top 5 Countries)")
        plt.show()

        # ===============================
        # 5. Advanced Visualizations
        # ===============================
        # Heatmap of Happiness by Country & Date
        pivot = df.pivot_table(
            values="Happiness_Score", index="Date", columns="Country", aggfunc="mean"
        )
        plt.figure(figsize=(12, 6))
        sns.heatmap(pivot.T, cmap="YlGnBu", cbar_kws={"label": "Happiness Score"})
        plt.title("Happiness Heatmap by Country and Date")
        plt.show()

        # FacetGrid: Happiness trend by sample countries
        sample_countries = df["Country"].value_counts().head(6).index
        g = sns.FacetGrid(
            df[df["Country"].isin(sample_countries)],
            col="Country",
            col_wrap=3,
            height=3.5,
        )
        g.map_dataframe(sns.lineplot, x="Date", y="Happiness_Score")
        g.set_titles("{col_name}")
        g.set_axis_labels("Date", "Happiness Score")
        g.set_xticklabels(rotation=90)
        plt.show()


# ==========================
# 🚀 Menu-driven interaction
# ==========================
def main():
    file_path = "global_happiness_report.csv"
    analyzer = HappinessDataAnalysis(file_path)

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
        print("\n--- Happiness Data Analysis Menu ---")
        for k, v in menu.items():
            print(f"{k}. {v[0]}")
        choice = int(input("Enter your choice: "))

        if choice == 0:
            print("Exiting...")
            delay(rand(1,6))
        elif choice in menu:
            menu[choice][1]()
        else:
            print("Invalid choice!")


if __name__ == "__main__":
    main()
