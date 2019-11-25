import numpy as np
import pandas as pd
import datetime


now = datetime.datetime.now()
start_time = datetime.datetime.strptime("2016-01-01", "%Y-%m-%d")


def timepreprocessing(string) -> np.ndarray:
    """
    :param string: string of date time
    :return: 5-dimensional numpy array corresponds to time
    for return
    0: check is holiday
    1, 2: day and month
    3, 4: hours

    """
    datetimeObject = datetime.datetime.strptime(string, "%Y-%m-%d %H:%M:%S")
    weekday = datetimeObject.weekday()
    """

    """
    ret = np.zeros(5)
    # mon 0, tues 1, ... sat 5, sun 6
    if weekday == 5 or weekday == 6: # is it holiday?
        ret[0] = 0
    else:
        ret[0] = 1
    delta = datetimeObject - start_time
    days_diff = delta.days
    ret[1] = np.cos(days_diff/ 365.25 * 2 * np.pi)
    ret[2] = np.sin(days_diff/ 365.25 * 2 * np.pi)

    hour = datetimeObject.timetuple().tm_hour
    ret[3] = np.cos(hour/24 * 2 * np.pi)
    ret[4] = np.sin(hour/24 * 2 * np.pi)
    return ret


def time_parser(data) -> np.ndarray:
    """
    :param data: this is time parts of the data; (N, 1) shape array where N is the number of data
    :return: numpy ndarray for whole data preprocessed (N, 5) shape array
    """
    ret = []
    for d in data:
        ret.append(timepreprocessing(d))
    return np.array(ret)


if __name__ == "__main__":
    csv_data = pd.read_csv("energydata_complete.csv", header=0)
    time = csv_data["date"].values
    print(time_parser(time)[0])
