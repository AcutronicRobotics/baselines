# from plot import loader, stick
import matplotlib
import matplotlib.pyplot as plt
import csv
from collections import defaultdict
import numpy as np

columns = defaultdict(list)

#matplotlib inline
matplotlib.rcParams.update({'font.size': 16})

color_defaults = [
    '#1f77b4',  # muted blue
    '#ff7f0e',  # safety orange
    '#2ca02c',  # cooked asparagus green
    '#d62728',  # brick red
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf'  # blue-teal
]
lines = []
names = []

label = 'ACKTR'


with open('/tmp/rosrl/GazeboModularScara3DOFv3Env/acktr/monitor/progress.csv') as f:
        reader = csv.DictReader(f) # read rows into a dictionary format
        for row in reader: # read a row as {column1: value1, column2: value2,...}
            for (k,v) in row.items(): # go over each column name and value
                columns[k].append(v) # append the value into the appropriate list
                                     # based on column name k

# print(columns['loss_vf_loss'])
# print(columns['loss_pol_surr'])
# print(np.asarray(columns['EpRewMean']))
# print(np.asarray(columns['EpRewSEM']))


color = color_defaults[0]
y_mean = np.asarray(list(map(float,columns['EpRewMean'])))
y_std = np.asarray(list(map(float,columns['EpRewSEM'])))

# x = np.asarray(list(map(float, columns['EVAfter'])))
x = np.linspace(0, 1e6, y_std.size, endpoint=True)


# color = colors[i]
y_upper = y_mean + y_std
y_lower = y_mean - y_std
# f2 = interp1d(y_upper, y_upper, kind='cubic')
plt.fill_between(
    x, list(y_lower), list(y_upper), interpolate=True, facecolor=color, linewidth=0.0, alpha=0.3
)

line = plt.plot(x, list(y_mean), label=label, color=color, rasterized=True)

lines.append(line[0])
names.append(label)

plot_name = 'Scara 3DoF'
plt.legend(lines,names, loc=4)
plt.xlim([0,1000000])
plt.xlabel("Number of Timesteps")
plt.ylabel("Episode Reward")
plt.title(plot_name)
plt.xticks([200000, 400000, 600000, 800000, 1000000], ["200K", "400K", "600K", "800K", "1M"])
plt.show()
