import numpy as np
import pandas as pd
import datetime


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


def binary(data):
    """
    :param data: light data. Note that light data is multiplication of 10
    :return: binary encoding of data
    """
    ret = np.zeros(3)
    data = data // 10
    ret[2] = (data  //4 ) % 2
    ret[1] = (data // 2) % 2
    ret[0] = data % 2
    return ret


def lightParser(light):
    ret = []
    for l in light:
        ret.append(binary(l))
    return np.array(ret)


def standardize(data):
    mu = np.mean(data)
    dev = np.std(data)
    return (data - mu) / dev


def appliancePreprocessing(data):
    data = standardize(data)
    ret = []
    for d in data:
        d = np.tanh(d)
        ret.append(d)
    return np.expand_dims(np.array(ret), -1)


def load_and_preprocessing():
    csv_data = pd.read_csv("energydata_complete.csv", header=0)
    npdata = csv_data.values
    time = npdata[:, 0]
    appliance = npdata[:, 1]
    light = npdata[:, 2]

    # Note that rv1 rv2  is just random variable
    other = npdata[:, 3: -2]

    preprocessed = []
    preprocessed.append(time_parser(time))
    preprocessed.append(appliancePreprocessing(appliance))
    preprocessed.append(lightParser(light))
    for c in range(other.shape[1]):
        column = other[:, c]
        preprocessed.append(np.expand_dims(standardize(column), axis=-1))
    return np.hstack(preprocessed)


COLNAME = ["Holiday", "Day And Month (cos)", "Day And Month (sin)", "Hour And Minute(cos)", "Hour And Minute(sin)",
          "Appliances",
           "lights0", "light1", "light2",
            "T1", "RH_1",
           "T2", "RH_2", "T3", "RH_3", "T4", "RH_4", "T5", "RH_5", "T6",
           "RH_6", "T7", "RH_7", "T8", "RH_8", "T9", "RH_9", "T_out",
           "Press_mm_hg", "RH_out", "Windspeed", "Visibility", "Tdewpoint"]


class toOriginalAppliance(object):
    """
    function which maps preprocessed appliance data to original appliance data
    This is built by class object because it has local constants whose names are very popular;
    """
    MEAN = 97.6949581960983
    DEV = 102.52229296483618

    def __call__(self, form):
        form = form * self.DEV + self.MEAN
        return np.arctanh(form)


if __name__ == "__main__":
    preprocessed = load_and_preprocessing()
    rework = pd.DataFrame(data=preprocessed, columns=COLNAME)
    rework.to_csv("preprocessed.csv", header=True, index=False)