# modules we'll use
import pandas as pd
import numpy as np
import seaborn as sns
import datetime

# read in our data
landslides = pd.read_csv("../datasets/catalog.csv")

# set seed for reproducibility
np.random.seed(0)

print(landslides['date'].dtype)

landslides['date_parsed'] = pd.to_datetime(landslides['date'], format="%m/%d/%y")
print(landslides['date_parsed'].dtype)


day_of_month_landslides = landslides['date_parsed'].dt.day
day_of_month_landslides = day_of_month_landslides.dropna()


sns.distplot(day_of_month_landslides, kde=False, bins=31)

