import pandas as pd
import numpy as np
import random

file_name = "Q1_air_quality.csv"

try:
    np.random.seed(321)
    random.seed(321)

    n_rows = 18000

    record_ids = [f"AQ_{str(i).zfill(6)}" for i in range(1, n_rows + 1)]
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

    usa_cities = [
        "New York",
        "Los Angeles",
        "Chicago",
        "Houston",
        "Phoenix",
        "Philadelphia",
        "San Antonio",
        "San Diego",
        "Dallas",
        "San Jose",
        "Austin",
        "Jacksonville",
        "Fort Worth",
        "Columbus",
        "Indianapolis",
        "Charlotte",
        "San Francisco",
        "Seattle",
        "Denver",
        "Washington",
    ]
    india_cities = [
        "Mumbai",
        "Delhi",
        "Bangalore",
        "Hyderabad",
        "Ahmedabad",
        "Chennai",
        "Kolkata",
        "Surat",
        "Pune",
        "Jaipur",
        "Lucknow",
        "Kanpur",
        "Nagpur",
        "Indore",
        "Thane",
        "Bhopal",
        "Visakhapatnam",
        "Patna",
        "Vadodara",
        "Ghaziabad",
    ]
    brazil_cities = [
        "Sao Paulo",
        "Rio de Janeiro",
        "Brasilia",
        "Salvador",
        "Fortaleza",
        "Belo Horizonte",
        "Manaus",
        "Curitiba",
        "Recife",
        "Goiania",
        "Porto Alegre",
        "Belem",
        "Guarulhos",
        "Campinas",
        "Sao Luis",
        "Sao Goncalo",
        "Maceio",
        "Duque de Caxias",
        "Natal",
        "Teresina",
    ]
    uk_cities = [
        "London",
        "Birmingham",
        "Manchester",
        "Liverpool",
        "Leeds",
        "Newcastle",
        "Sheffield",
        "Bristol",
        "Nottingham",
        "Leicester",
        "Coventry",
        "Hull",
        "Bradford",
        "Cardiff",
        "Belfast",
        "Glasgow",
        "Edinburgh",
        "Southampton",
        "Portsmouth",
        "Brighton",
    ]
    france_cities = [
        "Paris",
        "Marseille",
        "Lyon",
        "Toulouse",
        "Nice",
        "Nantes",
        "Strasbourg",
        "Montpellier",
        "Bordeaux",
        "Lille",
        "Rennes",
        "Reims",
        "Le Havre",
        "Saint-Etienne",
        "Toulon",
        "Grenoble",
        "Dijon",
        "Angers",
        "Nimes",
        "Villeurbanne",
    ]
    germany_cities = [
        "Berlin",
        "Hamburg",
        "Munich",
        "Cologne",
        "Frankfurt",
        "Stuttgart",
        "Dusseldorf",
        "Dortmund",
        "Essen",
        "Leipzig",
        "Bremen",
        "Dresden",
        "Hannover",
        "Nuremberg",
        "Duisburg",
        "Bochum",
        "Wuppertal",
        "Bielefeld",
        "Bonn",
        "Mannheim",
    ]
    italy_cities = [
        "Rome",
        "Milan",
        "Naples",
        "Turin",
        "Palermo",
        "Genoa",
        "Bologna",
        "Florence",
        "Bari",
        "Catania",
        "Venice",
        "Verona",
        "Messina",
        "Padua",
        "Trieste",
        "Taranto",
        "Brescia",
        "Prato",
        "Modena",
        "Reggio Calabria",
    ]
    spain_cities = [
        "Madrid",
        "Barcelona",
        "Valencia",
        "Seville",
        "Zaragoza",
        "Malaga",
        "Murcia",
        "Palma",
        "Las Palmas",
        "Bilbao",
        "Alicante",
        "Cordoba",
        "Valladolid",
        "Vigo",
        "Gijon",
        "Hospitalet",
        "La Coruna",
        "Granada",
        "Vitoria",
        "Elche",
    ]
    russia_cities = [
        "Moscow",
        "Saint Petersburg",
        "Novosibirsk",
        "Yekaterinburg",
        "Kazan",
        "Nizhny Novgorod",
        "Chelyabinsk",
        "Samara",
        "Omsk",
        "Rostov",
        "Ufa",
        "Krasnoyarsk",
        "Voronezh",
        "Perm",
        "Volgograd",
        "Krasnodar",
        "Saratov",
        "Tyumen",
        "Tolyatti",
        "Izhevsk",
    ]
    china_cities = [
        "Beijing",
        "Shanghai",
        "Guangzhou",
        "Shenzhen",
        "Chengdu",
        "Chongqing",
        "Tianjin",
        "Nanjing",
        "Wuhan",
        "Xi'an",
        "Hangzhou",
        "Dongguan",
        "Foshan",
        "Shenyang",
        "Harbin",
        "Qingdao",
        "Dalian",
        "Jinan",
        "Zhengzhou",
        "Changsha",
    ]
    japan_cities = [
        "Tokyo",
        "Yokohama",
        "Osaka",
        "Nagoya",
        "Sapporo",
        "Kobe",
        "Kyoto",
        "Fukuoka",
        "Kawasaki",
        "Saitama",
        "Hiroshima",
        "Sendai",
        "Kitakyushu",
        "Chiba",
        "Sakai",
        "Niigata",
        "Hamamatsu",
        "Kumamoto",
        "Sagamihara",
        "Okayama",
    ]
    sk_cities = [
        "Seoul",
        "Busan",
        "Incheon",
        "Daegu",
        "Daejeon",
        "Gwangju",
        "Ulsan",
        "Suwon",
        "Changwon",
        "Goyang",
        "Yongin",
        "Seongnam",
        "Cheongju",
        "Ansan",
        "Jeonju",
        "Anyang",
        "Pohang",
        "Bucheon",
        "Gimhae",
        "Masan",
    ]
    canada_cities = [
        "Toronto",
        "Montreal",
        "Vancouver",
        "Calgary",
        "Edmonton",
        "Ottawa",
        "Winnipeg",
        "Quebec City",
        "Hamilton",
        "Kitchener",
        "London",
        "Victoria",
        "Halifax",
        "Oshawa",
        "Windsor",
        "Saskatoon",
        "Regina",
        "St. John's",
        "Barrie",
        "Kelowna",
    ]
    australia_cities = [
        "Sydney",
        "Melbourne",
        "Brisbane",
        "Perth",
        "Adelaide",
        "Gold Coast",
        "Newcastle",
        "Canberra",
        "Wollongong",
        "Sunshine Coast",
        "Hobart",
        "Geelong",
        "Townsville",
        "Cairns",
        "Darwin",
        "Toowoomba",
        "Ballarat",
        "Bendigo",
        "Albury",
        "Launceston",
    ]
    mexico_cities = [
        "Mexico City",
        "Guadalajara",
        "Monterrey",
        "Puebla",
        "Tijuana",
        "León",
        "Juárez",
        "Zapopan",
        "Nezahualcóyotl",
        "Cancún",
        "Mérida",
        "Chihuahua",
        "San Luis Potosí",
        "Aguascalientes",
        "Hermosillo",
        "Saltillo",
        "Mexicali",
        "Culiacán",
        "Acapulco",
        "Morelia",
    ]
    sa_cities = [
        "Johannesburg",
        "Cape Town",
        "Durban",
        "Pretoria",
        "Port Elizabeth",
        "East London",
        "Bloemfontein",
        "Pietermaritzburg",
        "Kimberley",
        "Polokwane",
        "Nelspruit",
        "Rustenburg",
        "Welkom",
        "Newcastle",
        "George",
        "Midrand",
        "Centurion",
        "Vereeniging",
        "Soweto",
        "Pietersburg",
    ]
    turkey_cities = [
        "Istanbul",
        "Ankara",
        "Izmir",
        "Bursa",
        "Adana",
        "Gaziantep",
        "Konya",
        "Antalya",
        "Kayseri",
        "Mersin",
        "Eskisehir",
        "Diyarbakir",
        "Denzli",
        "Samsun",
        "Malatya",
        "Kahramanmaras",
        "Erzurum",
        "Van",
        "Batman",
        "Elazig",
    ]
    iran_cities = [
        "Tehran",
        "Mashhad",
        "Isfahan",
        "Karaj",
        "Tabriz",
        "Shiraz",
        "Qom",
        "Ahvaz",
        "Kermanshah",
        "Urmia",
        "Rasht",
        "Zahedan",
        "Hamadan",
        "Kerman",
        "Yazd",
        "Arak",
        "Ardabil",
        "Bandar Abbas",
        "Sanandaj",
        "Qazvin",
    ]

    country_cities = {
        "USA": usa_cities,
        "India": india_cities,
        "Brazil": brazil_cities,
        "UK": uk_cities,
        "France": france_cities,
        "Germany": germany_cities,
        "Italy": italy_cities,
        "Spain": spain_cities,
        "Russia": russia_cities,
        "China": china_cities,
        "Japan": japan_cities,
        "South Korea": sk_cities,
        "Canada": canada_cities,
        "Australia": australia_cities,
        "Mexico": mexico_cities,
        "South Africa": sa_cities,
        "Turkey": turkey_cities,
        "Iran": iran_cities,
    }

    selected_countries = []
    selected_cities = []
    for _ in range(n_rows):
        country = random.choice(countries)
        city = random.choice(country_cities[country])
        selected_countries.append(country)
        selected_cities.append(city)

    months = [1, 2, 3]
    days = list(range(1, 32))
    dates = []
    for _ in range(n_rows):
        month = random.choice(months)
        day = random.choice([d for d in days if not (month == 2 and d > 28)])
        dates.append(f"2025-{month:02d}-{day:02d}")

    pm25 = np.random.lognormal(3.0, 0.8, n_rows)
    pm25 = np.clip(np.round(pm25, 1), 5, 500)

    pm10 = pm25 * np.random.uniform(1.2, 2.5, n_rows)
    pm10 = np.clip(np.round(pm10, 1), 10, 600)

    no2 = np.random.lognormal(2.5, 0.7, n_rows)
    no2 = np.clip(np.round(no2, 1), 5, 200)

    so2 = np.random.lognormal(1.8, 0.6, n_rows)
    so2 = np.clip(np.round(so2, 1), 2, 100)

    co = np.random.lognormal(0.8, 0.5, n_rows)
    co = np.clip(np.round(co, 1), 0.1, 20)

    o3 = np.random.lognormal(2.2, 0.6, n_rows)
    o3 = np.clip(np.round(o3, 1), 10, 180)

    temperature = np.random.normal(15, 10, n_rows)
    temperature = np.clip(np.round(temperature, 1), -10, 40)

    humidity = np.random.normal(65, 20, n_rows)
    humidity = np.clip(np.round(humidity, 1), 10, 100)

    wind_speed = np.random.lognormal(1.5, 0.5, n_rows)
    wind_speed = np.clip(np.round(wind_speed, 1), 0, 30)

    aqi = np.round(
        pm25 * 0.4 + pm10 * 0.2 + no2 * 0.15 + so2 * 0.1 + co * 0.08 + o3 * 0.07, 0
    )
    aqi = np.clip(aqi, 0, 500)

    df = pd.DataFrame(
        {
            "Record_ID": record_ids,
            "Country": selected_countries,
            "City": selected_cities,
            "Date": dates,
            "PM2_5": pm25,
            "PM10": pm10,
            "NO2": no2,
            "SO2": so2,
            "CO": co,
            "O3": o3,
            "Temperature_C": temperature,
            "Humidity": humidity,
            "Wind_Speed_kmh": wind_speed,
            "AQI": aqi,
        }
    )

    duplicate_count = random.randint(100, 200)
    duplicate_indices = random.sample(range(n_rows), duplicate_count)
    duplicates = df.iloc[duplicate_indices].copy()
    df = pd.concat([df, duplicates], ignore_index=True)

    numeric_columns = [
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
    empty_count = random.randint(1000, 3000)
    empty_indices = random.sample(range(len(df)), empty_count)
    for idx in empty_indices:
        col = random.choice(numeric_columns)
        df.at[idx, col] = np.nan

    df = df.sample(frac=1, random_state=321).reset_index(drop=True)
except BaseException as e:
    print(f"Error occurred: {e}")
else:
    df.to_csv(file_name, index=False)
    print(
        f"Synthetic air quality data generated and saved as a CSV file named as {file_name}"
    )
finally:
    print("Data generation process completed.")
