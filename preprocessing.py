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
    4, 5: minute
    """
    datetimeObject = datetime.datetime.strptime(string, "%Y-%m-%d %H:%M:%S")
    weekday = datetimeObject.weekday()
    """

    """
    ret = np.zeros(7)
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

    minuite = datetimeObject.timetuple().tm_min
    ret[5] = np.cos(minuite/60 * 2 * np.pi)
    ret[6] = np.sin(minuite/60 * 2 * np.pi)
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
    ret = np.zeros(4)
    ret[3] = int(data >= 30)
    ret[2] = int (data == 20)
    ret[1] = int(data == 10)
    ret[0] = int(data == 0)
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
    data = data / 10
    data = standardize(data)
    ret = []
    for d in data:
        d = np.tanh(d)
        ret.append(d)
    return np.expand_dims(np.array(ret), -1)


def load_and_preprocessing():
    csv_data = pd.read_csv("energydata_complete.csv", header=0)
    print(csv_data.dtypes)

    time = csv_data["date"].values
    appliance = csv_data["Appliances"].values
    light = csv_data["lights"].values

    # Note that rv1 rv2  is just random variable
    preprocessed = []
    preprocessed.append(time_parser(time))

    preprocessed.append(appliancePreprocessing(appliance))
    np.save("appliances", appliancePreprocessing(appliance))
    preprocessed.append(lightParser(light))
    np.save("lights", (lightParser(light)))

    """
    Why these are ugly? 
    Because direct calling values makes numpy array with dtype object. 
    There must be better way, but don't care about it a bit.... 
    
    """

    npfile = []
    npfile.append(time_parser(time))
    for c in  ["T1", "RH_1", "T2", "RH_2", "T3", "RH_3", "T4", "RH_4", "T5", "RH_5", "T6",
                "RH_6", "T7", "RH_7", "T8", "RH_8", "T9", "RH_9", "T_out",
                "Press_mm_hg", "RH_out", "Windspeed", "Visibility", "Tdewpoint"]:
        column = csv_data[c].values
        preprocessed.append(np.expand_dims(standardize(column), axis=-1))
        npfile.append(np.expand_dims(standardize(column), axis=-1))
    np.save("X_data", np.hstack(npfile))
    return np.hstack(preprocessed)


COLNAME = ["Holiday", "Day And Month (cos)", "Day And Month (sin)",
           "Hour(cos)", "Hour(sin)", "Minute(cos)", "Minute(sin)",
           "Appliances", "lights0", "light1", "light2", "light3",
           "T1", "RH_1", "T2", "RH_2", "T3", "RH_3", "T4", "RH_4", "T5", "RH_5", "T6",
           "RH_6", "T7", "RH_7", "T8", "RH_8", "T9", "RH_9", "T_out",
           "Press_mm_hg", "RH_out", "Windspeed", "Visibility", "Tdewpoint"]


class toOriginalAppliance(object):
    """
    function which maps preprocessed appliance data to original appliance data
    This is built by class object because it has local constants whose names are very popular;
    """
    MEAN = 9.76949581960983
    DEV = 10.252229296483618

    def __call__(self, form):
        form = form * self.DEV + self.MEAN
        return np.arctanh(form)


if __name__ == "__main__":
    preprocessed = load_and_preprocessing()
    rework = pd.DataFrame(data=preprocessed, columns=COLNAME)
    rework.to_csv("preprocessed.csv", header=True, index=False)