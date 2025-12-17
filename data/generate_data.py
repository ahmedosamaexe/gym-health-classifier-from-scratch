import numpy as np
import pandas as pd

np.random.seed(42)
samples_per_class = 5000
rows = []

def generate_person(is_healthy):
    age = np.random.randint(18, 60)
    height = np.random.randint(155, 195)

    if is_healthy:
        bmi = np.random.uniform(18.5, 24.9)
        activity = np.random.randint(3, 6)
        label = 1
    else:
        bmi = np.random.choice([
            np.random.uniform(15, 18.4),
            np.random.uniform(25, 38)
        ])
        activity = np.random.randint(1, 4)
        label = 0

    weight = bmi * (height / 100) ** 2
    return [age, height, round(weight, 1), activity, label]

for _ in range(samples_per_class):
    rows.append(generate_person(True))
    rows.append(generate_person(False))

df = pd.DataFrame(rows, columns=[
    "age", "height_cm", "weight_kg", "activity_level", "healthy"
])

df.to_csv("data.csv", index=False)
print(df["healthy"].value_counts())
