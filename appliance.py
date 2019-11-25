import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == "__main__":
    csv_data = pd.read_csv("energydata_complete.csv", header=0)
    appliance  = csv_data["Appliances"].values
    mu = np.mean(appliance)
    sigma = np.std(appliance)
    stdzation = (appliance - mu)/sigma
    plt.hist(stdzation)
    plt.show()
