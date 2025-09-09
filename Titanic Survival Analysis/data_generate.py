import pandas as pd
import numpy as np
import random

file_name = "titanic_survival_dataset.csv"

try:
    np.random.seed(123)
    random.seed(123)

    n_rows = 15000

    passenger_ids = list(range(1, n_rows + 1))
    pclass = np.random.choice([1, 2, 3], n_rows, p=[0.15, 0.25, 0.6])

    male_first = [
        "James",
        "John",
        "Robert",
        "William",
        "Charles",
        "George",
        "Joseph",
        "Edward",
        "Henry",
        "Thomas",
        "Walter",
        "Frank",
        "Harry",
        "Albert",
        "Fred",
        "Arthur",
        "Samuel",
        "David",
        "Louis",
        "Richard",
    ]
    female_first = [
        "Mary",
        "Anna",
        "Elizabeth",
        "Margaret",
        "Ruth",
        "Helen",
        "Florence",
        "Dorothy",
        "Ethel",
        "Alice",
        "Edith",
        "Marie",
        "Catherine",
        "Grace",
        "Mildred",
        "Frances",
        "Rose",
        "Evelyn",
        "Gladys",
        "Lillian",
    ]
    last_names = [
        "Smith",
        "Johnson",
        "Brown",
        "Davis",
        "Wilson",
        "Miller",
        "Taylor",
        "Anderson",
        "Thomas",
        "Jackson",
        "White",
        "Harris",
        "Martin",
        "Thompson",
        "Garcia",
        "Martinez",
        "Robinson",
        "Clark",
        "Rodriguez",
        "Lewis",
    ]

    names = []
    for i in range(n_rows):
        if random.random() < 0.65:
            first = random.choice(male_first)
            title = random.choice(["Mr.", "Dr.", "Rev.", "Col.", "Major.", "Capt."])
        else:
            first = random.choice(female_first)
            title = random.choice(["Mrs.", "Miss.", "Ms.", "Lady.", "Countess.", "Mme."])
        last = random.choice(last_names)
        names.append(f"{title} {first} {last}")

    sex = np.random.choice(["male", "female"], n_rows, p=[0.65, 0.35])

    age = np.random.normal(29, 14, n_rows)
    age = np.clip(np.round(age), 0.5, 80)

    sibsp = np.random.poisson(0.5, n_rows)
    sibsp = np.clip(sibsp, 0, 8)

    parch = np.random.poisson(0.4, n_rows)
    parch = np.clip(parch, 0, 6)

    ticket_prefixes = [
        "A/5",
        "PC",
        "STON/O2",
        "C.A",
        "SOTON/OQ",
        "W./C",
        "SC/Paris",
        "CA",
        "SC/Paris",
        "F.C.C",
        "LINE",
        "PP",
        "SC/AH",
        "A/4",
        "A/S",
    ]
    tickets = []
    for i in range(n_rows):
        tickets.append(f"{random.choice(ticket_prefixes)} {random.randint(1000, 9999)}")

    fare = np.random.lognormal(2.5, 1.2, n_rows)
    fare = np.clip(np.round(fare, 2), 0, 512)

    cabins = []
    for i in range(n_rows):
        if random.random() < 0.3:
            deck = random.choice(["A", "B", "C", "D", "E", "F", "G"])
            cabin_num = random.randint(1, 150)
            cabins.append(f"{deck}{cabin_num}")
        else:
            cabins.append(np.nan)

    embarked = np.random.choice(["S", "C", "Q"], n_rows, p=[0.72, 0.19, 0.09])

    survival_prob = np.where(sex == "female", 0.74, 0.19)
    survival_prob = np.where(pclass == 1, survival_prob + 0.15, survival_prob)
    survival_prob = np.where(age < 16, survival_prob + 0.2, survival_prob)
    survival_prob = np.where(fare > 100, survival_prob + 0.1, survival_prob)
    survival_prob = np.clip(survival_prob, 0.05, 0.95)
    survived = np.random.binomial(1, survival_prob)

    df = pd.DataFrame(
        {
            "PassengerId": passenger_ids,
            "Survived": survived,
            "Pclass": pclass,
            "Name": names,
            "Sex": sex,
            "Age": age,
            "SibSp": sibsp,
            "Parch": parch,
            "Ticket": tickets,
            "Fare": fare,
            "Cabin": cabins,
            "Embarked": embarked,
        }
    )

    duplicate_count = random.randint(100, 200)
    duplicate_indices = random.sample(range(n_rows), duplicate_count)
    duplicates = df.iloc[duplicate_indices].copy()
    df = pd.concat([df, duplicates], ignore_index=True)

    numeric_columns = ["Age", "SibSp", "Parch", "Fare"]
    empty_count = random.randint(1000, 3000)
    empty_indices = random.sample(range(len(df)), empty_count)
    for idx in empty_indices:
        col = random.choice(numeric_columns)
        df.at[idx, col] = np.nan

    df = df.sample(frac=1, random_state=123).reset_index(drop=True)
except BaseException as e:
    print(f"An error occurred: {e}")
else:
    df.to_csv(file_name, index=False)
