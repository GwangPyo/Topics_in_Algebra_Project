import numpy as np
import pandas as pd


if __name__ == "__main__":
    csv_data = pd.read_csv("energydata_complete.csv", header=0)
    time = csv_data["date"]

