import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from time import sleep as delay
from random import randint as rand


# ===============================
# 1. Generate Data
# ===============================
def generate_data():
    if os.path.exists("data_generate.py") and not os.path.exists("Q1_air_quality.csv"):
        os.system("python data_generate.py")
        print("✅ Data generated successfully.")
    elif not os.path.exists("Q1_air_quality.csv"):
        print("❌ data already exists!")
    elif not os.path.exists("data_generate.py"):
        print("❌ data_generate.py not found!")


# ===============================
# 2. Load Data
# ===============================
def load_data(file_path):
    if not os.path.exists(file_path):
        print("⚠️ Data file not found!")
        return None
    df = pd.read_csv(file_path)
    print("✅ Data loaded successfully.")
    print("Shape:", df.shape)
    return df


# ===============================
# 3. Basic Info
# ===============================
def basic_info(df):
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
def handle_missing_values(df):
    if df is None:
        print("⚠️ Data not loaded.")
        return df
    numeric_cols = df.select_dtypes(include=[np.number]).columns
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
    return df


# ===============================
# 5. All Analysis
# ===============================
def all_analysis(df):
    if df is None:
        print("⚠️ Data not loaded.")
        return
    print("\n--- All Analysis ---")

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

    # Basic Info
    print("Dataset Shape:", df.shape)
    print("\nColumn Data Types:\n", df.dtypes)
    print("\nMissing Values:\n", df.isna().sum())
    print("\nSummary Statistics:\n", df[numeric_cols].describe(include="all"))
    print("\nMissing Values (%):\n", (df.isna().sum() / len(df)) * 100)

    # AQI Analysis
    print("\nOverall Average AQI:", df["AQI"].mean())
    print(
        "\nAverage AQI by Country:\n",
        df.groupby("Country")["AQI"].mean().sort_values(ascending=False).head(10),
    )
    print(
        "\nAverage AQI by City:\n",
        df.groupby("City")["AQI"].mean().sort_values(ascending=False).head(10),
    )
    df["Month"] = pd.to_datetime(df["Date"]).dt.month
    print("\nAverage AQI by Month:\n", df.groupby("Month")["AQI"].mean())

    # Pollutant Analysis
    print(
        "\nMean Pollutant Concentrations:\n",
        df[numeric_cols[:-3]].mean().sort_values(ascending=False),
    )
    print(
        "\nMaximum Recorded Pollutant Levels:\n",
        df[numeric_cols[:-3]].max().sort_values(ascending=False),
    )
    country_means = df.groupby("Country")[numeric_cols].mean()
    pollutant_vs_aqi = country_means.drop(columns="AQI").corrwith(country_means["AQI"])
    print(
        "\nCorrelation of Pollutants with AQI (by Country averages):\n",
        pollutant_vs_aqi,
    )

    # Weather & AQI Relationships
    print(
        "\nAQI by Temperature Range:\n",
        df.groupby(pd.cut(df["Temperature_C"], bins=5))["AQI"].mean(),
    )
    print(
        "\nAQI by Humidity Range:\n",
        df.groupby(pd.cut(df["Humidity"], bins=5))["AQI"].mean(),
    )
    print(
        "\nAQI by Wind Speed Range:\n",
        df.groupby(pd.cut(df["Wind_Speed_kmh"], bins=5))["AQI"].mean(),
    )

    # Correlation Matrix
    print("\nCorrelation Matrix:\n", df[numeric_cols].corr())

    # Derived Metrics
    df["PM_Ratio"] = df["PM2_5"] / df["PM10"].replace(0, np.nan)
    print("\nAverage PM2.5 to PM10 Ratio:", df["PM_Ratio"].mean())
    df["Pollution_Burden"] = df[["PM2_5", "PM10", "NO2", "SO2", "CO", "O3"]].sum(axis=1)
    print(
        "\nTop 5 Records with Highest Pollution Burden:\n",
        df.nlargest(5, "Pollution_Burden")[["Country", "City", "Pollution_Burden"]],
    )
    df["Extreme_AQI"] = (df["AQI"] > 100).astype(int)
    print("\nProportion of Extreme AQI Days (>100):", df["Extreme_AQI"].mean())
    df["Unhealthy_Day"] = (df["AQI"] > 50).astype(int)
    print(
        "\nTop 10 Countries by Proportion of Unhealthy Days:\n",
        df.groupby("Country")["Unhealthy_Day"]
        .mean()
        .sort_values(ascending=False)
        .head(10),
    )


# ===============================
# 6. All Visualizations
# ===============================
def all_visualizations(df):
    if df is None:
        print("⚠️ Data not loaded.")
        return

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

    # Pairplot
    sns.pairplot(df[numeric_cols].sample(500, random_state=42))
    plt.show()

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

    # Violin plots
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
    df = None

    menu = {
        1: ("Create Data", lambda: generate_data()),
        2: ("Load Data", lambda: globals().update(df := load_data(file_path))),
        3: ("Basic Info", lambda: basic_info(df)),
        4: ("Handle Missing Values", lambda: handle_missing_values(df)),
        5: ("All Analysis", lambda: all_analysis(df)),
        6: ("All Visualizations", lambda: all_visualizations(df)),
        0: ("Exit", None),
    }

    while True:
        print("\n--- Custom Data Analysis Menu (POP) ---")
        for k, v in menu.items():
            print(f"{k}. {v[0]}")
        choice = int(input("Enter your choice: "))

        if choice == 0:
            print("Exiting...")
            delay(rand(1, 3))
            break
        elif choice in menu:
            if choice == 2:
                df = load_data(file_path)
            else:
                menu[choice][1]()
        else:
            print("❌ Invalid choice!")


if __name__ == "__main__":
    main()
