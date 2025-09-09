import pandas as pd
import numpy as np
import random

file_name = "global_happiness_report.csv"

try:
    np.random.seed(789)
    random.seed(789)

    n_rows = 16000

    record_ids = [f"HAPPY_{str(i).zfill(6)}" for i in range(1, n_rows + 1)]
    countries = [
        "USA",
        "India",
        "Brazil",
        "UK",
        "France",
        "Germany",
        "Italy",
        "Spain",
        "Russia",
        "China",
        "Japan",
        "South Korea",
        "Canada",
        "Australia",
        "Mexico",
        "South Africa",
        "Turkey",
        "Iran",
    ]

    usa_states = [
        "California",
        "Texas",
        "Florida",
        "New York",
        "Illinois",
        "Pennsylvania",
        "Ohio",
        "Georgia",
        "North Carolina",
        "Michigan",
        "Washington",
        "Colorado",
        "Virginia",
        "Massachusetts",
        "Arizona",
    ]
    india_states = [
        "Maharashtra",
        "Tamil Nadu",
        "Kerala",
        "Karnataka",
        "Andhra Pradesh",
        "Uttar Pradesh",
        "Delhi",
        "West Bengal",
        "Rajasthan",
        "Gujarat",
        "Madhya Pradesh",
        "Bihar",
        "Punjab",
        "Haryana",
        "Odisha",
    ]
    brazil_states = [
        "Sao Paulo",
        "Rio de Janeiro",
        "Minas Gerais",
        "Bahia",
        "Parana",
        "Rio Grande do Sul",
        "Pernambuco",
        "Ceara",
        "Santa Catarina",
        "Goias",
        "Maranhao",
        "Para",
        "Amazonas",
        "Espirito Santo",
        "Paraiba",
    ]
    uk_regions = [
        "England",
        "Scotland",
        "Wales",
        "Northern Ireland",
        "London",
        "South East",
        "North West",
        "East of England",
        "West Midlands",
        "South West",
        "Yorkshire",
        "East Midlands",
        "North East",
    ]
    france_regions = [
        "Ile-de-France",
        "Auvergne-Rhone-Alpes",
        "Provence-Alpes-Cote d'Azur",
        "Hauts-de-France",
        "Grand Est",
        "Occitanie",
        "Pays de la Loire",
        "Brittany",
        "Normandy",
        "Nouvelle-Aquitaine",
        "Centre-Val de Loire",
        "Bourgogne-Franche-Comte",
        "Corsica",
    ]
    germany_states = [
        "North Rhine-Westphalia",
        "Bavaria",
        "Baden-Wurttemberg",
        "Lower Saxony",
        "Hesse",
        "Saxony",
        "Rhineland-Palatinate",
        "Berlin",
        "Schleswig-Holstein",
        "Hamburg",
        "Brandenburg",
        "Mecklenburg-Vorpommern",
        "Saarland",
        "Thuringia",
        "Saxony-Anhalt",
    ]
    italy_regions = [
        "Lombardy",
        "Lazio",
        "Campania",
        "Veneto",
        "Emilia-Romagna",
        "Piedmont",
        "Sicily",
        "Apulia",
        "Tuscany",
        "Calabria",
        "Liguria",
        "Marche",
        "Abruzzo",
        "Umbria",
        "Basilicata",
        "Molise",
        "Trentino",
        "Friuli",
        "Valle d'Aosta",
    ]
    spain_regions = [
        "Madrid",
        "Catalonia",
        "Andalusia",
        "Valencia",
        "Castile and Leon",
        "Basque Country",
        "Castilla-La Mancha",
        "Galicia",
        "Aragon",
        "Murcia",
        "Asturias",
        "Extremadura",
        "Balearic Islands",
        "Canary Islands",
        "Cantabria",
        "Navarre",
        "La Rioja",
    ]
    russia_regions = [
        "Moscow",
        "Saint Petersburg",
        "Moscow Oblast",
        "Krasnodar Krai",
        "Sverdlovsk Oblast",
        "Rostov Oblast",
        "Republic of Bashkortostan",
        "Republic of Tatarstan",
        "Chelyabinsk Oblast",
        "Novosibirsk Oblast",
        "Nizhny Novgorod Oblast",
        "Samara Oblast",
        "Krasnoyarsk Krai",
        "Irkutsk Oblast",
        "Volgograd Oblast",
    ]
    china_provinces = [
        "Hubei",
        "Guangdong",
        "Henan",
        "Zhejiang",
        "Hunan",
        "Anhui",
        "Jiangxi",
        "Jiangsu",
        "Chongqing",
        "Sichuan",
        "Shandong",
        "Hebei",
        "Fujian",
        "Shaanxi",
        "Guangxi",
        "Heilongjiang",
        "Yunnan",
        "Jilin",
        "Liaoning",
        "Shanxi",
    ]
    japan_prefectures = [
        "Tokyo",
        "Osaka",
        "Kanagawa",
        "Aichi",
        "Saitama",
        "Chiba",
        "Hyogo",
        "Hokkaido",
        "Fukuoka",
        "Kyoto",
        "Hiroshima",
        "Niigata",
        "Miyagi",
        "Nagano",
        "Gifu",
        "Ibaraki",
        "Shizuoka",
        "Okayama",
        "Kumamoto",
        "Tochigi",
    ]
    sk_provinces = [
        "Seoul",
        "Busan",
        "Incheon",
        "Daegu",
        "Daejeon",
        "Gwangju",
        "Ulsan",
        "Gyeonggi",
        "Gangwon",
        "Chungcheong",
        "Jeolla",
        "Gyeongsang",
        "Jeju",
    ]
    canada_provinces = [
        "Ontario",
        "Quebec",
        "British Columbia",
        "Alberta",
        "Manitoba",
        "Saskatchewan",
        "Nova Scotia",
        "New Brunswick",
        "Newfoundland and Labrador",
        "Prince Edward Island",
        "Yukon",
        "Northwest Territories",
        "Nunavut",
    ]
    australia_states = [
        "New South Wales",
        "Victoria",
        "Queensland",
        "Western Australia",
        "South Australia",
        "Tasmania",
        "Australian Capital Territory",
        "Northern Territory",
    ]
    mexico_states = [
        "Mexico City",
        "State of Mexico",
        "Jalisco",
        "Nuevo Leon",
        "Guanajuato",
        "Puebla",
        "Veracruz",
        "Baja California",
        "Chihuahua",
        "Sonora",
        "Tamaulipas",
        "Coahuila",
        "Michoacan",
        "Guerrero",
        "Oaxaca",
        "Chiapas",
        "Sinaloa",
        "Durango",
        "San Luis Potosi",
        "Zacatecas",
    ]
    sa_provinces = [
        "Gauteng",
        "Western Cape",
        "KwaZulu-Natal",
        "Eastern Cape",
        "Free State",
        "Mpumalanga",
        "North West",
        "Limpopo",
        "Northern Cape",
    ]
    turkey_provinces = [
        "Istanbul",
        "Ankara",
        "Izmir",
        "Bursa",
        "Antalya",
        "Konya",
        "Adana",
        "Gaziantep",
        "Kocaeli",
        "Mersin",
        "Kayseri",
        "Diyarbakir",
        "Hatay",
        "Manisa",
        "Samsun",
        "Balikesir",
        "Kahramanmaras",
        "Van",
        "Eskisehir",
        "Malatya",
    ]
    iran_provinces = [
        "Tehran",
        "Isfahan",
        "Razavi Khorasan",
        "Fars",
        "East Azerbaijan",
        "Mazandaran",
        "Alborz",
        "Kerman",
        "Gilan",
        "Golestan",
        "West Azerbaijan",
        "Kermanshah",
        "Lorestan",
        "Hormozgan",
        "Sistan and Baluchestan",
        "Qom",
        "Kurdistan",
        "Hamadan",
        "Yazd",
        "Ardabil",
    ]

    country_states = {
        "USA": usa_states,
        "India": india_states,
        "Brazil": brazil_states,
        "UK": uk_regions,
        "France": france_regions,
        "Germany": germany_states,
        "Italy": italy_regions,
        "Spain": spain_regions,
        "Russia": russia_regions,
        "China": china_provinces,
        "Japan": japan_prefectures,
        "South Korea": sk_provinces,
        "Canada": canada_provinces,
        "Australia": australia_states,
        "Mexico": mexico_states,
        "South Africa": sa_provinces,
        "Turkey": turkey_provinces,
        "Iran": iran_provinces,
    }

    selected_countries = []
    selected_states = []
    for _ in range(n_rows):
        country = random.choice(countries)
        state = random.choice(country_states[country])
        selected_countries.append(country)
        selected_states.append(state)

    years = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
    dates = []
    for _ in range(n_rows):
        year = random.choice(years)
        month = random.randint(1, 12)
        day = random.randint(1, 28)
        dates.append(f"{year}-{month:02d}-{day:02d}")

    happiness_score = np.random.uniform(2.5, 8.5, n_rows)
    happiness_score = np.round(happiness_score, 3)

    gdp_per_capita = np.random.lognormal(9.5, 0.8, n_rows)
    gdp_per_capita = np.round(gdp_per_capita, 2)

    social_support = np.random.beta(2, 1, n_rows)
    social_support = np.round(social_support * 2, 3)

    healthy_life_expectancy = np.random.normal(65, 12, n_rows)
    healthy_life_expectancy = np.clip(np.round(healthy_life_expectancy, 1), 45, 85)

    freedom_to_make_life_choices = np.random.beta(3, 2, n_rows)
    freedom_to_make_life_choices = np.round(freedom_to_make_life_choices * 1.5, 3)

    generosity = np.random.normal(0, 0.3, n_rows)
    generosity = np.clip(np.round(generosity, 3), -0.5, 0.5)

    perceptions_of_corruption = np.random.beta(2, 3, n_rows)
    perceptions_of_corruption = np.round(perceptions_of_corruption, 3)

    positive_affect = np.random.beta(3, 2, n_rows)
    positive_affect = np.round(positive_affect, 3)

    negative_affect = np.random.beta(2, 3, n_rows)
    negative_affect = np.round(negative_affect, 3)

    confidence_in_government = np.random.beta(2, 3, n_rows)
    confidence_in_government = np.round(confidence_in_government, 3)

    df = pd.DataFrame(
        {
            "Record_ID": record_ids,
            "Country": selected_countries,
            "State_Region": selected_states,
            "Date": dates,
            "Happiness_Score": happiness_score,
            "GDP_Per_Capita": gdp_per_capita,
            "Social_Support": social_support,
            "Healthy_Life_Expectancy": healthy_life_expectancy,
            "Freedom_To_Make_Life_Choices": freedom_to_make_life_choices,
            "Generosity": generosity,
            "Perceptions_Of_Corruption": perceptions_of_corruption,
            "Positive_Affect": positive_affect,
            "Negative_Affect": negative_affect,
            "Confidence_In_Government": confidence_in_government,
        }
    )

    duplicate_count = random.randint(100, 200)
    duplicate_indices = random.sample(range(n_rows), duplicate_count)
    duplicates = df.iloc[duplicate_indices].copy()
    df = pd.concat([df, duplicates], ignore_index=True)

    numeric_columns = [
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
    empty_count = random.randint(1000, 3000)
    empty_indices = random.sample(range(len(df)), empty_count)
    for idx in empty_indices:
        col = random.choice(numeric_columns)
        df.at[idx, col] = np.nan

    df = df.sample(frac=1, random_state=789).reset_index(drop=True)
except BaseException as e:
    print(f"An error occurred: {e}")
else:
    df.to_csv(file_name, index=False)
    print(f"Data successfully generated and saved as a CSV file named as \"{file_name}\"")
finally:
    print("Data generation process completed.")
