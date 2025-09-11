import pandas as pd
import numpy as np
import random

file_name = "Q1_stock_market.csv"

try:
    np.random.seed(555)
    random.seed(555)

    n_rows = 17500

    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "JPM", "JNJ", "V", "PG", "UNH", "HD", "MA", "DIS", "ADBE", "NFLX", "PYPL", "CRM", "INTC", "CSCO", "PEP", "ABT", "TMO", "AVGO", "QCOM", "TXN", "ACN", "HON", "IBM", "ORCL", "NKE", "PM", "LIN", "AMGN", "MDT", "UPS", "SBUX", "CAT", "MMM", "GS", "BA", "RTX", "GE", "F", "GM", "AMD", "MU", "ATVI", "EA"]
    company_names = {
        "AAPL": "Apple Inc.", "MSFT": "Microsoft Corporation", "GOOGL": "Alphabet Inc.", 
        "AMZN": "Amazon.com Inc.", "TSLA": "Tesla Inc.", "META": "Meta Platforms Inc.", 
        "NVDA": "NVIDIA Corporation", "JPM": "JPMorgan Chase & Co.", "JNJ": "Johnson & Johnson", 
        "V": "Visa Inc.", "PG": "Procter & Gamble Co.", "UNH": "UnitedHealth Group Inc.", 
        "HD": "Home Depot Inc.", "MA": "Mastercard Inc.", "DIS": "Walt Disney Co.", 
        "ADBE": "Adobe Inc.", "NFLX": "Netflix Inc.", "PYPL": "PayPal Holdings Inc.", 
        "CRM": "Salesforce Inc.", "INTC": "Intel Corporation", "CSCO": "Cisco Systems Inc.", 
        "PEP": "PepsiCo Inc.", "ABT": "Abbott Laboratories", "TMO": "Thermo Fisher Scientific Inc.", 
        "AVGO": "Broadcom Inc.", "QCOM": "Qualcomm Inc.", "TXN": "Texas Instruments Inc.", 
        "ACN": "Accenture plc", "HON": "Honeywell International Inc.", "IBM": "International Business Machines Corp.", 
        "ORCL": "Oracle Corporation", "NKE": "Nike Inc.", "PM": "Philip Morris International Inc.", 
        "LIN": "Linde plc", "AMGN": "Amgen Inc.", "MDT": "Medtronic plc", "UPS": "United Parcel Service Inc.", 
        "SBUX": "Starbucks Corporation", "CAT": "Caterpillar Inc.", "MMM": "3M Company", 
        "GS": "Goldman Sachs Group Inc.", "BA": "Boeing Co.", "RTX": "RTX Corporation", 
        "GE": "General Electric Co.", "F": "Ford Motor Co.", "GM": "General Motors Co.", 
        "AMD": "Advanced Micro Devices Inc.", "MU": "Micron Technology Inc.", 
        "ATVI": "Activision Blizzard Inc.", "EA": "Electronic Arts Inc."
    }

    sectors = {
        "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology", 
        "AMZN": "Consumer Cyclical", "TSLA": "Automotive", "META": "Communication Services", 
        "NVDA": "Technology", "JPM": "Financial Services", "JNJ": "Healthcare", 
        "V": "Financial Services", "PG": "Consumer Defensive", "UNH": "Healthcare", 
        "HD": "Consumer Cyclical", "MA": "Financial Services", "DIS": "Communication Services", 
        "ADBE": "Technology", "NFLX": "Communication Services", "PYPL": "Financial Services", 
        "CRM": "Technology", "INTC": "Technology", "CSCO": "Technology", 
        "PEP": "Consumer Defensive", "ABT": "Healthcare", "TMO": "Healthcare", 
        "AVGO": "Technology", "QCOM": "Technology", "TXN": "Technology", 
        "ACN": "Technology", "HON": "Industrials", "IBM": "Technology", 
        "ORCL": "Technology", "NKE": "Consumer Cyclical", "PM": "Consumer Defensive", 
        "LIN": "Basic Materials", "AMGN": "Healthcare", "MDT": "Healthcare", 
        "UPS": "Industrials", "SBUX": "Consumer Cyclical", "CAT": "Industrials", 
        "MMM": "Industrials", "GS": "Financial Services", "BA": "Industrials", 
        "RTX": "Industrials", "GE": "Industrials", "F": "Consumer Cyclical", 
        "GM": "Consumer Cyclical", "AMD": "Technology", "MU": "Technology", 
        "ATVI": "Communication Services", "EA": "Communication Services"
    }

    selected_symbols = []
    selected_companies = []
    selected_sectors = []
    for _ in range(n_rows):
        symbol = random.choice(symbols)
        selected_symbols.append(symbol)
        selected_companies.append(company_names[symbol])
        selected_sectors.append(sectors[symbol])

    months = [1, 2, 3]
    days = list(range(1, 32))
    dates = []
    for _ in range(n_rows):
        month = random.choice(months)
        day = random.choice([d for d in days if not (month == 2 and d > 28)])
        dates.append(f"2025-{month:02d}-{day:02d}")

    base_prices = {
        "AAPL": 180, "MSFT": 380, "GOOGL": 140, "AMZN": 150, "TSLA": 220, 
        "META": 320, "NVDA": 450, "JPM": 170, "JNJ": 155, "V": 250, 
        "PG": 145, "UNH": 500, "HD": 330, "MA": 400, "DIS": 95, 
        "ADBE": 520, "NFLX": 550, "PYPL": 65, "CRM": 230, "INTC": 40, 
        "CSCO": 50, "PEP": 170, "ABT": 110, "TMO": 550, "AVGO": 1200, 
        "QCOM": 140, "TXN": 165, "ACN": 320, "HON": 200, "IBM": 180, 
        "ORCL": 120, "NKE": 100, "PM": 95, "LIN": 400, "AMGN": 280, 
        "MDT": 85, "UPS": 160, "SBUX": 95, "CAT": 230, "MMM": 105, 
        "GS": 380, "BA": 220, "RTX": 85, "GE": 130, "F": 12, 
        "GM": 40, "AMD": 150, "MU": 85, "ATVI": 90, "EA": 130
    }

    open_prices = []
    high_prices = []
    low_prices = []
    close_prices = []
    volumes = []

    for symbol in selected_symbols:
        base_price = base_prices[symbol]
        daily_volatility = np.random.uniform(0.01, 0.04)
        
        open_price = base_price * (1 + np.random.uniform(-0.02, 0.02))
        daily_change = np.random.normal(0, daily_volatility)
        close_price = open_price * (1 + daily_change)
        
        high_price = max(open_price, close_price) * (1 + np.random.uniform(0.005, 0.03))
        low_price = min(open_price, close_price) * (1 - np.random.uniform(0.005, 0.03))
        
        volume = np.random.lognormal(14, 1.2)
        
        open_prices.append(np.round(open_price, 2))
        high_prices.append(np.round(high_price, 2))
        low_prices.append(np.round(low_price, 2))
        close_prices.append(np.round(close_price, 2))
        volumes.append(int(volume * 1000))

    market_caps = []
    for symbol, close in zip(selected_symbols, close_prices):
        base_cap = {
            "AAPL": 2.8, "MSFT": 2.5, "GOOGL": 1.8, "AMZN": 1.6, "TSLA": 0.7, 
            "META": 1.0, "NVDA": 1.1, "JPM": 0.5, "JNJ": 0.4, "V": 0.5, 
            "PG": 0.35, "UNH": 0.45, "HD": 0.35, "MA": 0.4, "DIS": 0.18, 
            "ADBE": 0.25, "NFLX": 0.24, "PYPL": 0.08, "CRM": 0.22, "INTC": 0.18, 
            "CSCO": 0.2, "PEP": 0.23, "ABT": 0.19, "TMO": 0.22, "AVGO": 0.5, 
            "QCOM": 0.16, "TXN": 0.15, "ACN": 0.2, "HON": 0.13, "IBM": 0.16, 
            "ORCL": 0.32, "NKE": 0.16, "PM": 0.15, "LIN": 0.2, "AMGN": 0.15, 
            "MDT": 0.11, "UPS": 0.14, "SBUX": 0.11, "CAT": 0.12, "MMM": 0.06, 
            "GS": 0.13, "BA": 0.14, "RTX": 0.12, "GE": 0.14, "F": 0.05, 
            "GM": 0.06, "AMD": 0.24, "MU": 0.09, "ATVI": 0.07, "EA": 0.04
        }[symbol] * 1e12
        market_caps.append(int(base_cap * (close / base_prices[symbol])))

    pe_ratios = np.random.normal(25, 8, n_rows)
    pe_ratios = np.clip(np.round(pe_ratios, 1), 8, 60)

    dividend_yields = np.random.uniform(0, 4.5, n_rows)
    dividend_yields = np.round(dividend_yields, 2)

    rsi = np.random.normal(50, 15, n_rows)
    rsi = np.clip(np.round(rsi, 1), 0, 100)

    df = pd.DataFrame({
        "Record_ID": [f"STK_{str(i).zfill(6)}" for i in range(1, len(selected_symbols) + 1)],
        "Symbol": selected_symbols,
        "Company_Name": selected_companies,
        "Sector": selected_sectors,
        "Date": dates,
        "Open_Price": open_prices,
        "High_Price": high_prices,
        "Low_Price": low_prices,
        "Close_Price": close_prices,
        "Volume": volumes,
        "Market_Cap": market_caps,
        "PE_Ratio": pe_ratios,
        "Dividend_Yield": dividend_yields,
        "RSI": rsi
    })

    duplicate_count = random.randint(100, 200)
    duplicate_indices = random.sample(range(n_rows), duplicate_count)
    duplicates = df.iloc[duplicate_indices].copy()
    df = pd.concat([df, duplicates], ignore_index=True)

    numeric_columns = ["Open_Price", "High_Price", "Low_Price", "Close_Price", "Volume", "Market_Cap", "PE_Ratio", "Dividend_Yield", "RSI"]
    empty_count = random.randint(1000, 3000)
    empty_indices = random.sample(range(len(df)), empty_count)
    for idx in empty_indices:
        col = random.choice(numeric_columns)
        df.at[idx, col] = np.nan

    df = df.sample(frac=1, random_state=555).reset_index(drop=True)
except BaseException as e:
    print(f"An error occurred: {e}")
else:
    df.to_csv(file_name, index=False)
finally:
    print(f"Data generation completed. Dataset saved to {file_name}.")
    print("Execution finished.")