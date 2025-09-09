import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from time import sleep as delay
from random import randint as rand
import os

# Global dataframe
df = None
file_path = "global_happiness_report.csv"


# 1. Generate Data
def generate_data():
    if os.path.exists("data_generate.py") and not os.path.exists(file_path):
        os.system("python data_generate.py")
        print("✅ Data generated successfully.")
    elif os.path.exists(file_path):
        print("⚠️ Data already exists!")
    else:
        print("❌ data_generate.py not found!")


# 2. Load Data
def load_data():
    global df
    if not os.path.exists(file_path):
        print("⚠️ Data file not found. Generating new data...")
        generate_data()

    df = pd.read_csv(file_path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    print("✅ Data loaded successfully.")
    print("Shape:", df.shape)


# 3. Basic Info
def basic_info():
    global df
    print("\n--- Basic Info ---")
    print(df.info())
    print("\nMissing Values:\n", df.isna().sum())
    print("\nMissing Values (%):\n", (df.isna().sum() / len(df)) * 100)


# 4. Handle Missing Values
def handle_missing_values():
    global df
    numeric_cols = [
        "Confirmed_Cases",
        "Deaths",
        "Recovered",
        "Active_Cases",
        "Tests_Conducted",
        "Vaccination_Rate",
        "Hospitalization_Rate",
        "ICU_Cases",
    ]
    while True:
        print("\n--- Handling Missing Values ---")
        print("1. Fill numeric columns with aggregate function (mean/median)")
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
                    print(f"✅ Missing values filled using {func}")
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


# 5. All Analysis
def all_analysis():
    global df
    if df is None:
        print("⚠️ Data not loaded. Please load data first.")
        return

    print("\n--- All Analysis ---")
    print("Dataset Shape:", df.shape)
    print("\nColumn Data Types:\n", df.dtypes)
    print("\nMissing Values:\n", df.isna().sum())

    numeric_cols = [
        "Confirmed_Cases",
        "Deaths",
        "Recovered",
        "Active_Cases",
        "Tests_Conducted",
        "Vaccination_Rate",
        "Hospitalization_Rate",
        "ICU_Cases",
    ]
    print("\nSummary Statistics:\n", df[numeric_cols].describe())

    print(
        "\nTop 10 Countries by Confirmed Cases:\n",
        df.groupby("Country")["Confirmed_Cases"].sum().nlargest(10),
    )

    print(
        "\nTop 10 Countries by Deaths:\n",
        df.groupby("Country")["Deaths"].sum().nlargest(10),
    )

    print(
        "\nTop 10 Countries by Avg Vaccination Rate:\n",
        df.groupby("Country")["Vaccination_Rate"].mean().nlargest(10),
    )

    time_series = df.groupby("Date")[
        ["Confirmed_Cases", "Deaths", "Recovered", "Active_Cases"]
    ].sum()
    time_series["Daily_New_Cases"] = time_series["Confirmed_Cases"].diff()
    print("\nTime Series (first 10 rows):\n", time_series.head(10))

    print("\nCorrelation Matrix:\n", df[numeric_cols].corr())

    print(
        "\nTop 10 Countries by Avg ICU Cases:\n",
        df.groupby("Country")["ICU_Cases"].mean().nlargest(10),
    )

    print(
        "\nTop 10 Countries by Avg Hospitalization Rate:\n",
        df.groupby("Country")["Hospitalization_Rate"].mean().nlargest(10),
    )

    print(
        "\nTop 10 States/Regions by Confirmed Cases:\n",
        df.groupby(["Country", "State_Region"])["Confirmed_Cases"].sum().nlargest(10),
    )

    df["Death_Rate"] = df["Deaths"] / df["Confirmed_Cases"].replace(0, np.nan)
    df["Recovery_Rate"] = df["Recovered"] / df["Confirmed_Cases"].replace(0, np.nan)
    print("\nGlobal Avg Death Rate:", df["Death_Rate"].mean())
    print("Global Avg Recovery Rate:", df["Recovery_Rate"].mean())


# 6. All Visualizations
def all_visualizations():
    global df
    if df is None:
        print("⚠️ Data not loaded. Please load data first.")
        return

    numeric_cols = [
        "Confirmed_Cases",
        "Deaths",
        "Recovered",
        "Active_Cases",
        "Tests_Conducted",
        "Vaccination_Rate",
        "Hospitalization_Rate",
        "ICU_Cases",
    ]

    for col in numeric_cols:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col].dropna(), bins=30, kde=True)
        plt.title(f"Distribution of {col}")
        plt.show()

    # More plots (boxplots, scatter, heatmaps, etc.)
    # ... same as in your OOP version


# ==========================
# 🚀 Menu-driven interaction
# ==========================
def main():
    menu = {
        1: ("Generate Data (Run data_generate.py)", generate_data),
        2: ("Load Data", load_data),
        3: ("Basic Info", basic_info),
        4: ("Handle Missing Values", handle_missing_values),
        5: ("All Analysis", all_analysis),
        6: ("All Visualizations", all_visualizations),
        0: ("Exit", None),
    }

    while True:
        print("\n--- COVID Data Analysis Menu ---")
        for k, v in menu.items():
            print(f"{k}. {v[0]}")
        choice = int(input("Enter your choice: "))

        match choice:
            case 0:
                print("Exiting...")
                delay(rand(1,6))
                break
            case _ if choice in menu:
                menu[choice][1]()
            case _:
                print("❌ Invalid choice!")


if __name__ == "__main__":
    main()
