from config.config import data_dir
from analyzer.analyzer import SSIXAnalyzer
from loader.loader import Loader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt

def get_day(created_at):
    return created_at[:10]

fig = plt.figure()
ax = plt.axes()

def get_timeline_count(tweets_file, keywords=None, timezones=None):
    # get dataframe
    l = Loader(tweets_file)
    data = l.get_dataframe()

    def valid_keyword(x):
        for keyword in keywords:
            if keyword in x:
                return True
        return False

    def valid_timezone(t):
        for timezone in timezones:
            if timezone in t:
                return True
        return False

    if not keywords is None:
        data = data[data["text"].apply(valid_keyword) == True]

    if not timezones is None:
        data = data[data["user_time_zone"].apply(valid_timezone) == True]

    print len(data.index)

    data["created_at"] = data["created_at"].astype("datetime64")
    a = data["created_at"].groupby(data["created_at"].dt.day).count()
    a = a.to_frame()
    return np.array(a.index), np.array(a.values)

def fill_with_zeros(x, y):
    """
        Elements in x and y don't necessarily have the same length.
        This function makes them compatible by inserting zeros at the right places.
    """
    # get all distinct x values
    x_full = set([])
    for xv in x:
        x_full |= set(xv)

    N = len(x_full)
    x_filled = np.array(list(x_full))
    print x_filled

    x_res = len(x)*[x_filled]
    y_res = []

    for xv, yv in zip(x, y):
        y_filled = np.zeros(N)
        for j in range(xv.size):
            for i in range(N): 
                if xv[j] == x_filled[i]:
                    y_filled[i] = yv[j]
                    break
        y_res.append(y_filled)

    return x_res, y_res
        

def plot_timelines(x, y, colors, labels, saveas=None):
    N = len(x)
    assert N == len(y), "x and y must be lists of same length"

    bar_width = 1.0/N
   
    for i in range(N):
        xk = x[i] + bar_width*(i - N/2 - 1)
        ax.bar(xk, y[i], width=bar_width, color=colors[i], label=labels[i])

    ax.set_xlabel("Date")
    ax.set_ylabel("Word frequency")
    plt.legend(loc="best")
    if not saveas is None:
        plt.savefig(saveas)

def plot_stacked_timelines(ax, x, y, colors, labels):
    """
        Plot stacked timelines to ax
    """
    N = len(x)
    assert N == len(y), "x and y must be lists of same length"
    assert N > 0, "x and y must not be empty"

    bar_width = 0.5 # bar width in plot

    ybottom = np.zeros(x[0].size) # needed for stacking
    for i in range(N):
        ax.bar(x[i], y[i], bottom=ybottom, width=bar_width, color=colors[i], label=labels[i])
        ybottom += y[i].flatten()
        print "Plotted ", labels[i]

    
def foreach_keyword(tweets_csv, keywords, timezones):
    """
        Filter for each keyword in keywords and matching timezones
    """
    x, y, labels = [], [], []
    for keyword in keywords:
        if keyword == "":
            xn, yn = get_timeline_count(tweets_csv, timezones)
            labels.append("All tweets")
        else:
            xn, yn = get_timeline_count(tweets_csv, [keyword], timezones)
            labels.append("Matching '{}'".format(keyword))
        x.append(xn)
        y.append(yn)

    return x, y, labels


def run_experiment(tweets_csv, keywords, timezoneslist, titles):
    for timezones, title in zip(timezoneslist, titles):
        fig = plt.figure()
        ax = plt.axes()
        if len(timezones) == 1 and timezones[0] == "":
            ax.set_title("All timezones")
        else:
            ax.set_title("Timezone: {}".format(title))

        x, y, labels = foreach_keyword(tweets_csv, keywords, timezones)
        xf, yf = fill_with_zeros(x, y)
        plot_stacked_timelines(ax, xf, yf, colors, labels)
    
        ax.set_xlabel("Date")
        ax.set_ylabel("Word frequency")
        plt.legend(loc="best")
        plt.savefig("frequency_stacked_{}.pdf".format(title))


# file with tweets
tweets_csv = data_dir + "brexit_all_2.csv"

keywords = ["strongerin", "ukineu", "ukip", "leave", "remain", "euref"]
timezones = [ ["London"],
              ["Europe", "London", "Berlin"],
              ["Berlin"] ]
titles = ["London", "Europe", "Berlin"]
colors = ["b", "g", "r", "m", "c", "y"]

run_experiment(tweets_csv, keywords, timezones, titles)
