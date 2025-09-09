import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from time import sleep as delay
from random import randint as rand


class CustomDataAnalysis:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None

    # 1. Generate Data
    def generate_data(self):
        if os.path.exists("data_generate.py") and not os.path.exists(
            "Q1_air_quality.csv"
        ):
            os.system("python data_generate.py")
            print("✅ Data generated successfully.")
        elif not os.path.exists("Q1_air_quality.csv"):
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
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
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
        print("\n--- All Analysis ---")
        # ===============================
        # 1. Basic Info
        # ===============================
        print("Dataset Shape:", df.shape)
        print("\nColumn Data Types:\n", df.dtypes)
        print("\nMissing Values:\n", df.isna().sum())

        numeric_cols = [
            "PM2_5",
            "PM10",
            "NO2",
            "SO2",
            "CO",
            "O3",
            "Temperature_C",
            "Humidity",
            "Wind_Speed_kmh",
            "AQI",
        ]

        print("\nSummary Statistics:\n", df[numeric_cols].describe(include="all"))

        missing_percent = (df.isna().sum() / len(df)) * 100
        print("\nMissing Values (%):\n", missing_percent)

        # ===============================
        # 2. AQI Analysis
        # ===============================
        avg_aqi = df["AQI"].mean()
        print("\nOverall Average AQI:", avg_aqi)

        aqi_by_country = (
            df.groupby("Country")["AQI"].mean().sort_values(ascending=False)
        )
        print("\nAverage AQI by Country:\n", aqi_by_country.head(10))

        aqi_by_city = df.groupby("City")["AQI"].mean().sort_values(ascending=False)
        print("\nAverage AQI by City:\n", aqi_by_city.head(10))

        df["Month"] = pd.to_datetime(df["Date"]).dt.month
        aqi_by_month = df.groupby("Month")["AQI"].mean()
        print("\nAverage AQI by Month:\n", aqi_by_month)

        # ===============================
        # 3. Pollutant Analysis
        # ===============================
        pollutant_means = df[numeric_cols[:-3]].mean().sort_values(ascending=False)
        print("\nMean Pollutant Concentrations:\n", pollutant_means)

        pollutant_max = df[numeric_cols[:-3]].max().sort_values(ascending=False)
        print("\nMaximum Recorded Pollutant Levels:\n", pollutant_max)

        # ✅ Fixed correlation calculation
        country_means = df.groupby("Country")[numeric_cols].mean()
        aqi_means = country_means["AQI"]
        pollutant_means_only = country_means.drop(columns="AQI")
        pollutant_vs_aqi = pollutant_means_only.corrwith(aqi_means)

        print(
            "\nCorrelation of Pollutants with AQI (by Country averages):\n",
            pollutant_vs_aqi,
        )

        # ===============================
        # 4. Weather & AQI Relationships
        # ===============================
        avg_temp_aqi = df.groupby(pd.cut(df["Temperature_C"], bins=5))["AQI"].mean()
        print("\nAQI by Temperature Range:\n", avg_temp_aqi)

        avg_humidity_aqi = df.groupby(pd.cut(df["Humidity"], bins=5))["AQI"].mean()
        print("\nAQI by Humidity Range:\n", avg_humidity_aqi)

        avg_wind_aqi = df.groupby(pd.cut(df["Wind_Speed_kmh"], bins=5))["AQI"].mean()
        print("\nAQI by Wind Speed Range:\n", avg_wind_aqi)

        # ===============================
        # 5. Correlations
        # ===============================
        corr_matrix = df[numeric_cols].corr()
        print("\nCorrelation Matrix:\n", corr_matrix)

        # ===============================
        # 6. Derived Metrics
        # ===============================
        # Example: PM2.5 / PM10 ratio
        df["PM_Ratio"] = df["PM2_5"] / df["PM10"].replace(0, np.nan)
        print("\nAverage PM2.5 to PM10 Ratio:", df["PM_Ratio"].mean())

        # Example: Pollution Burden Index (sum of pollutants)
        df["Pollution_Burden"] = df[["PM2_5", "PM10", "NO2", "SO2", "CO", "O3"]].sum(
            axis=1
        )
        print(
            "\nTop 5 Records with Highest Pollution Burden:\n",
            df.nlargest(5, "Pollution_Burden")[["Country", "City", "Pollution_Burden"]],
        )

        # Example: Extreme Pollution Flag
        df["Extreme_AQI"] = (df["AQI"] > 100).astype(int)
        extreme_aqi_rate = df["Extreme_AQI"].mean()
        print("\nProportion of Extreme AQI Days (>100):", extreme_aqi_rate)

        # Example: Healthy vs Unhealthy Days (AQI threshold 50)
        df["Unhealthy_Day"] = (df["AQI"] > 50).astype(int)
        unhealthy_rate = (
            df.groupby("Country")["Unhealthy_Day"].mean().sort_values(ascending=False)
        )
        print(
            "\nTop 10 Countries by Proportion of Unhealthy Days:\n",
            unhealthy_rate.head(10),
        )

    # 5. All Visualizations
    def all_visualizations(self):
        if self.df is None:
            print("⚠️ Data not loaded.")
            return
        df = self.df
        numeric_cols = [
            "PM2_5",
            "PM10",
            "NO2",
            "SO2",
            "CO",
            "O3",
            "Temperature_C",
            "Humidity",
            "Wind_Speed_kmh",
            "AQI",
        ]

        # ===============================
        # 1. Univariate Analysis
        # ===============================

        # Histograms
        for col in numeric_cols:
            plt.figure(figsize=(6, 4))
            sns.histplot(df[col], bins=30, kde=True)
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

        # Scatterplots (pollutants vs AQI)
        for col in numeric_cols[:-1]:
            plt.figure(figsize=(6, 4))
            sns.scatterplot(data=df, x=col, y="AQI", alpha=0.5)
            plt.title(f"{col} vs AQI")
            plt.show()

        # Correlation Heatmap
        plt.figure(figsize=(10, 6))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Heatmap")
        plt.show()

        # Pairplot (sample for speed)
        sns.pairplot(df[numeric_cols].sample(500, random_state=42))
        plt.show()

        # ===============================
        # 3. Categorical Analysis
        # ===============================

        # AQI distribution by Country
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df, x="Country", y="AQI")
        plt.xticks(rotation=90)
        plt.title("AQI Distribution by Country")
        plt.show()

        # AQI distribution by City (top 15)
        top_cities = df["City"].value_counts().head(15).index
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df[df["City"].isin(top_cities)], x="City", y="AQI")
        plt.xticks(rotation=90)
        plt.title("AQI Distribution in Top 15 Cities")
        plt.show()

        # ===============================
        # 4. Advanced Visualizations
        # ===============================

        # Violin plots for pollutants by country (sampled for clarity)
        for col in ["PM2_5", "PM10", "NO2", "SO2", "CO", "O3"]:
            plt.figure(figsize=(12, 6))
            sns.violinplot(data=df.sample(2000, random_state=42), x="Country", y=col)
            plt.xticks(rotation=90)
            plt.title(f"{col} Distribution by Country")
            plt.show()

        # Heatmap: Average AQI by Country & Month
        df["Month"] = pd.to_datetime(df["Date"]).dt.month
        pivot = df.pivot_table(
            values="AQI", index="Country", columns="Month", aggfunc="mean"
        )
        plt.figure(figsize=(12, 6))
        sns.heatmap(pivot, annot=False, cmap="YlGnBu", cbar_kws={"label": "Avg AQI"})
        plt.title("Average AQI by Country & Month")
        plt.show()

        # FacetGrid: Distribution of PM2.5 by Country
        g = sns.FacetGrid(
            df.sample(3000, random_state=42), col="Country", col_wrap=4, height=3
        )
        g.map_dataframe(sns.histplot, x="PM2_5", bins=20, kde=True)
        g.set_axis_labels("PM2.5", "Count")
        plt.show()


# ==========================
# 🚀 Menu-driven interaction
# ==========================
def main():
    file_path = "Q1_air_quality.csv"
    analyzer = CustomDataAnalysis(file_path)

    menu = {
        1: ("Create Data", analyzer.generate_data),
        2: ("Load Data", analyzer.load_data),
        3: ("Basic Info", analyzer.basic_info),
        4: ("Handle Missing Values", analyzer.handle_missing_values),
        5: ("All Analysis", analyzer.all_analysis),
        6: ("All Visualizations", analyzer.all_visualizations),
        0: ("Exit", None),
    }

    while True:
        print("\n--- Custom Data Analysis Menu ---")
        for k, v in menu.items():
            print(f"{k}. {v[0]}")
        choice = int(input("Enter your choice: "))

        if choice == 0:
            print("Exiting...")
            delay(rand(1, 3))
            break
        elif choice in menu:
            menu[choice][1]()
        else:
            print("❌ Invalid choice!")


if __name__ == "__main__":
    main()
