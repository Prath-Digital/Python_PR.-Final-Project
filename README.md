
# 🐍 Python Data Analysis Final Project 📊

## 📝 Overview
This repository contains multiple data analysis and visualization projects, each focusing on a different real-world dataset. The projects demonstrate data cleaning, exploratory data analysis (EDA), and visualization using Python libraries such as Pandas, Matplotlib, Seaborn, and others. Each project is organized in its own folder with code, data, and requirements.

## 📁 Project Structure


- 🦠 **COVID-19 Data Analysis and Visualization**
	- Analyze the spread of COVID-19 over time, examining trends in cases, recoveries, and deaths across different countries or regions. Visualize the impact of government interventions.
	- **Dataset:** Johns Hopkins University COVID-19 Dataset, Our World in Data COVID-19 Dataset
	- **Files:**
		- `main_oop.py`, `main_pop.py`, `main.ipynb`, `data_generate.py`, `covid19_global_data.csv`, `requirements.txt`

- 😊 **Global Happiness Report Analysis**
	- Analyze the World Happiness Report to understand factors contributing to happiness in different countries. Visualize correlations between happiness scores and variables such as GDP per capita, social support, and life expectancy.
	- **Dataset:** World Happiness Report Dataset (Kaggle)
	- **Files:**
		- `main_oop.py`, `main_pop.py`, `main.ipynb`, `data_generate.py`, `global_happiness_report.csv`, `requirements.txt`

- 🚢 **Titanic Survival Analysis**
	- Perform EDA on the Titanic dataset to understand factors influencing passenger survival. Create visualizations for survival rates by class, gender, age, etc.
	- **Dataset:** Titanic Dataset (Kaggle)
	- **Files:**
		- `main_oop.py`, `main_pop.py`, `main.ipynb`, `data_generate.py`, `titanic_survival_dataset.csv`, `requirements.txt`

- 🌫️ **Air Quality Analysis**
	- Analyze air quality data from various locations to understand pollution levels over time. Visualize trends in air quality indices and their relationship with weather or public health metrics.
	- **Dataset:** UCI Machine Learning Repository Air Quality Dataset, OpenAQ Global Air Quality Data
	- **Files:**
		- `main_oop.py`, `main_pop.py`, `main.ipynb`, `data_generate.py`, `Q1_air_quality.csv`, `requirements.txt`

- 💹 **Stock Market Analysis**
	- Analyze historical stock market data to identify trends and patterns in stock prices. Visualize stock performance against various indicators such as moving averages or trading volume.
	- **Dataset:** Yahoo Finance Historical Stock Prices, yfinance library, Kaggle Stock Market Datasets
	- **Files:**
		- `main_oop.py`, `main_pop.py`, `main.ipynb`, `data_generate.py`, `Q1_stock_market.csv`, `requirements.txt`

## 🛠️ Tools & Libraries
- 🐍 Python 3.10+
- 🐼 Pandas (data manipulation)
- 📊 Matplotlib (visualization)
- 🖼️ Seaborn (visualization)
- 🔢 NumPy (numerical calculations, some projects)

## 📋 Instructions

1. ⚙️ **Setup**
	- Clone this repository.
	- Navigate to the desired project folder.
	- Install dependencies using the provided `requirements.txt` file:
	  ```bash
	  pip install -r requirements.txt
	  ```

2. ▶️ **Running the Code**
	- Each project contains both script files (`main_oop.py`, `main_pop.py`) and a Jupyter notebook (`main.ipynb`).
	- You can run the scripts directly or open the notebook for an interactive analysis.

3. 🗂️ **Data**
	- Datasets are included in each project folder. You may also download updated datasets from the links provided in the project descriptions or from [Kaggle](https://www.kaggle.com/).

4. 📝 **Assumptions**
	- Any assumptions made during analysis are documented within the code or notebooks.

5. 🚫 **No Plagiarism**
	- All code is original. Do not copy from unauthorized sources.

## 📤 Submission
Once your project is complete, upload it to your GitHub repository and submit the link as instructed. Ensure your repository is well-organized and includes all required files.

---


## 🖥️ Console Inputs & Outputs / Notebook Output Types

Below are the typical console inputs and outputs, as well as the types of outputs you can expect from the Jupyter Notebooks for each project:

### 🦠 COVID-19 Data Analysis and Visualization
- **Console Inputs:**
	- Menu-driven: Enter your choice (e.g., 1. Generate Data, 2. Load Data, etc.)
	- Aggregate function input: mean/median (for missing value handling)
- **Console Outputs:**
	- Data loading status, shape, info, missing values, summary statistics, analysis results, and error/warning messages.
- **Notebook Output Types:**
	- Standard output (text), tables, HTML, images (plots), and interactive charts.

### 😊 Global Happiness Report Analysis
- **Console Inputs:**
	- Menu-driven: Enter your choice (e.g., 1. Generate Data, 2. Load Data, etc.)
	- Aggregate function input: mean/median (for missing value handling)
- **Console Outputs:**
	- Data loading status, shape, info, missing values, summary statistics, country/region analysis, and error/warning messages.
- **Notebook Output Types:**
	- Standard output (text), tables, HTML, images (plots), and interactive charts.

### 🚢 Titanic Survival Analysis
- **Console Inputs:**
	- Menu-driven: Enter your choice (e.g., 1. Create Dataset, 2. Load Data, etc.)
	- Aggregate function input: mean/median (for missing value handling)
- **Console Outputs:**
	- Data loading status, shape, info, missing values, summary statistics, survival analysis, and error/warning messages.
- **Notebook Output Types:**
	- Standard output (text), tables, HTML, images (plots), and interactive charts.

### 🌫️ Air Quality Analysis
- **Console Inputs:**
	- Menu-driven: Enter your choice (e.g., 1. Create Data, 2. Load Data, etc.)
	- Aggregate function input: mean/median (for missing value handling)
- **Console Outputs:**
	- Data loading status, shape, info, missing values, summary statistics, AQI and pollutant analysis, and error/warning messages.
- **Notebook Output Types:**
	- Standard output (text), tables, HTML, images (plots), and interactive charts.

### 💹 Stock Market Analysis
- **Console Inputs:**
	- Menu-driven: Enter your choice (e.g., 1. Generate Data, 2. Load Data, etc.)
	- Aggregate function input: mean/median (for missing value handling)
- **Console Outputs:**
	- Data loading status, shape, info, missing values, summary statistics, stock/sector analysis, and error/warning messages.
- **Notebook Output Types:**
	- Standard output (text), tables, HTML, images (plots), and interactive charts.

**Notebook Output Details:**
- All notebooks include code cells that produce outputs such as:
	- `stdout` (printed text, tables)
	- `text/html` (rendered tables, interactive widgets)
	- `image/png` (plots, charts)
	- `stderr` (error messages, if any)

---
**Note:** For dataset sources and further details, refer to the project folders and comments in the code. Good luck with your project work! 🚀
