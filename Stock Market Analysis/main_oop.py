import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from time import sleep as delay
from random import randint as rand


class StockDataAnalysis:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None

    # 1. Generate Data
    def generate_data(self, file_location):
        if os.path.exists("data_generate.py") and not os.path.exists(file_location):
            os.system("python data_generate.py")
            print("✅ Data generated successfully.")
        elif not os.path.exists(file_location):
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
            "Open_Price",
            "High_Price",
            "Low_Price",
            "Close_Price",
            "Volume",
            "Market_Cap",
            "PE_Ratio",
            "Dividend_Yield",
            "RSI",
        ]

        print("\nSummary Statistics:\n", df[numeric_cols].describe(include="all"))

        missing_percent = (df.isna().sum() / len(df)) * 100
        print("\nMissing Values (%):\n", missing_percent)

        # ===============================
        # 2. Stock Price Analysis
        # ===============================
        avg_close = df["Close_Price"].mean()
        print("\nOverall Average Close Price:", avg_close)

        close_by_sector = (
            df.groupby("Sector")["Close_Price"].mean().sort_values(ascending=False)
        )
        print("\nAverage Close Price by Sector:\n", close_by_sector.head(10))

        close_by_symbol = (
            df.groupby("Symbol")["Close_Price"].mean().sort_values(ascending=False)
        )
        print("\nAverage Close Price by Symbol:\n", close_by_symbol.head(10))

        df["Month"] = pd.to_datetime(df["Date"]).dt.month
        close_by_month = df.groupby("Month")["Close_Price"].mean()
        print("\nAverage Close Price by Month:\n", close_by_month)

        # ===============================
        # 3. Market Metrics Analysis
        # ===============================
        market_means = (
            df[["Volume", "Market_Cap", "PE_Ratio", "Dividend_Yield", "RSI"]]
            .mean()
            .sort_values(ascending=False)
        )
        print("\nMean Market Metrics:\n", market_means)

        market_max = (
            df[["Volume", "Market_Cap", "PE_Ratio", "Dividend_Yield", "RSI"]]
            .max()
            .sort_values(ascending=False)
        )
        print("\nMaximum Recorded Market Metrics:\n", market_max)

        # Correlation of metrics with Close Price (by sector averages)
        sector_means = df.groupby("Sector")[numeric_cols].mean()
        close_means = sector_means["Close_Price"]
        metrics_only = sector_means.drop(columns="Close_Price")
        metrics_vs_close = metrics_only.corrwith(close_means)

        print(
            "\nCorrelation of Metrics with Close Price (by Sector averages):\n",
            metrics_vs_close,
        )

        # ===============================
        # 4. Ranges & Relationships
        # ===============================
        avg_vol_close = df.groupby(pd.qcut(df["Volume"], q=5))["Close_Price"].mean()
        print("\nAverage Close Price by Volume Quintile:\n", avg_vol_close)

        avg_pe_close = df.groupby(pd.qcut(df["PE_Ratio"], q=5))["Close_Price"].mean()
        print("\nAverage Close Price by PE Ratio Quintile:\n", avg_pe_close)

        avg_rsi_close = df.groupby(pd.cut(df["RSI"], bins=[0, 30, 70, 100]))[
            "Close_Price"
        ].mean()
        print("\nAverage Close Price by RSI Range:\n", avg_rsi_close)

        # ===============================
        # 5. Correlations
        # ===============================
        corr_matrix = df[numeric_cols].corr()
        print("\nCorrelation Matrix:\n", corr_matrix)

        # ===============================
        # 6. Derived Metrics
        # ===============================
        # Example: Daily Price Range
        df["Daily_Range"] = df["High_Price"] - df["Low_Price"]
        print("\nAverage Daily Price Range:", df["Daily_Range"].mean())

        # Example: Volatility Ratio (Range / Close)
        df["Volatility_Ratio"] = df["Daily_Range"] / df["Close_Price"].replace(
            0, np.nan
        )
        print(
            "\nTop 5 Records with Highest Volatility:\n",
            df.nlargest(5, "Volatility_Ratio")[
                ["Symbol", "Sector", "Date", "Volatility_Ratio"]
            ],
        )

        # Example: Overbought/Oversold Flag using RSI
        df["Overbought"] = (df["RSI"] > 70).astype(int)
        df["Oversold"] = (df["RSI"] < 30).astype(int)
        overbought_rate = df["Overbought"].mean()
        oversold_rate = df["Oversold"].mean()
        print("\nProportion of Overbought Days:", overbought_rate)
        print("Proportion of Oversold Days:", oversold_rate)

        # Example: Dividend Presence Flag
        df["Has_Dividend"] = (df["Dividend_Yield"] > 0).astype(int)
        dividend_rate = (
            df.groupby("Sector")["Has_Dividend"].mean().sort_values(ascending=False)
        )
        print(
            "\nSectors with Highest Proportion of Dividend Stocks:\n",
            dividend_rate.head(10),
        )

    # 6. All Visualizations
    def all_visualizations(self):
        if self.df is None:
            print("⚠️ Data not loaded.")
            return
        df = self.df
        # ===============================
        # Numeric Columns
        # ===============================
        numeric_cols = [
            "Open_Price", "High_Price", "Low_Price", "Close_Price",
            "Volume", "Market_Cap", "PE_Ratio", "Dividend_Yield", "RSI"
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

        # Scatterplots (numeric vs Close_Price)
        for col in [c for c in numeric_cols if c != "Close_Price"]:
            plt.figure(figsize=(6, 4))
            sns.scatterplot(data=df, x=col, y="Close_Price", alpha=0.5)
            plt.title(f"{col} vs Close_Price")
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

        # Close Price distribution by Sector
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df, x="Sector", y="Close_Price")
        plt.xticks(rotation=90)
        plt.title("Close Price Distribution by Sector")
        plt.show()

        # Close Price distribution by Top 15 Symbols
        top_symbols = df["Symbol"].value_counts().head(15).index
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df[df["Symbol"].isin(top_symbols)], x="Symbol", y="Close_Price")
        plt.xticks(rotation=90)
        plt.title("Close Price Distribution in Top 15 Stocks")
        plt.show()


        # ===============================
        # 4. Advanced Visualizations
        # ===============================

        # Violin plots for key metrics by Sector (sample for clarity)
        for col in ["Close_Price", "Volume", "PE_Ratio", "RSI"]:
            plt.figure(figsize=(12, 6))
            sns.violinplot(data=df.sample(2000, random_state=42), x="Sector", y=col)
            plt.xticks(rotation=90)
            plt.title(f"{col} Distribution by Sector")
            plt.show()

        # Heatmap: Average Close Price by Sector & Month
        df["Month"] = pd.to_datetime(df["Date"]).dt.month
        pivot = df.pivot_table(values="Close_Price", index="Sector", columns="Month", aggfunc="mean")
        plt.figure(figsize=(12, 6))
        sns.heatmap(pivot, annot=False, cmap="YlGnBu", cbar_kws={"label": "Avg Close Price"})
        plt.title("Average Close Price by Sector & Month")
        plt.show()

        # FacetGrid: RSI distribution by Sector
        g = sns.FacetGrid(df.sample(3000, random_state=42), col="Sector", col_wrap=4, height=3)
        g.map_dataframe(sns.histplot, x="RSI", bins=20, kde=True)
        g.set_axis_labels("RSI", "Count")
        plt.show()



# ==========================
# 🚀 Menu-driven interaction
# ==========================
def main():
    file_path = "Q1_stock_market.csv"
    analyzer = StockDataAnalysis(file_path)

    menu = {
        1: ("Generate Data", analyzer.generate_data),
        2: ("Load Data", analyzer.load_data),
        3: ("Basic Info", analyzer.basic_info),
        4: ("Handle Missing Values", analyzer.handle_missing_values),
        5: ("All Analysis", analyzer.all_analysis),
        6: ("All Visualizations", analyzer.all_visualizations),
        0: ("Exit", None),
    }

    while True:
        print("\n--- Stock Data Analysis Menu ---")
        for k, v in menu.items():
            print(f"{k}. {v[0]}")
        choice = int(input("Enter your choice: "))

        match choice:
            case 0:
                print("Exiting...")
                delay(rand(1, 3))
                break
            case _ if choice in menu:
                match choice:
                    case 1:
                        menu[choice][1](file_path)
                    case _:
                        menu[choice][1]()
            case _:
                print("❌ Invalid choice!")


if __name__ == "__main__":
    main()
