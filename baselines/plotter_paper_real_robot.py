# from plot import loader, stick
import matplotlib
import matplotlib.pyplot as plt
import csv
from collections import defaultdict
import numpy as np

from scipy.signal import savgol_filter



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

# label = 'PPO1'

def plot_results(plot_name, all_values, labels, smooth=True):
    lines = []
    names = []

    for i in range(len(all_values)):
        y_mean = []
        y_std = []
        y_upper = []
        y_lower = []
        columns = defaultdict(list)
        print(all_values[i])
        with open(all_values[i]) as f:
                reader = csv.DictReader(f) # read rows into a dictionary format
                for row in reader: # read a row as {column1: value1, column2: value2,...}
                    for (k,v) in row.items(): # go over each column name and value
                        if v is '':
                            v = 'nan'
                        columns[k].append(v) # append the value into the appropriate list
                                             # based on column name k

        # print(columns['loss_vf_loss'])
        # print(columns['loss_pol_surr'])
        # print(np.asarray(columns['EpRewMean']))
        # print(np.asarray(columns['EpRewSEM']))

        color = color_defaults[i]

        # if i is 0:
        #     color = color_defaults[i]
        # else:
        #     color = color_defaults[i+1]
        # if i > 2 and i < 5:
        #     y_mean = np.asarray(list(map(float,columns['EpRewMean100'])))
        #     y_std = np.asarray(list(map(float,columns['EpRewMean100'])))
        # else:
        y_mean = np.asarray(list(map(float,columns['RMSE'])))
        y_std = np.std(y_mean)

        # now calculate the Eucledian Distance between target and the positions recorded.
        points = [np.asarray(list(map(float,columns['x']))), np.asarray(list(map(float,columns['y']))), np.asarray(list(map(float,columns['z'])))]

        # print(points[0])

        # target
        EE_POS_TGT = np.transpose(np.asmatrix([0.3305805, -0.1326121, 0]))
        # print(EE_POS_TGT.shape)

        # for
        distance =  np.subtract(points, EE_POS_TGT)

        print("distance: ", distance)


        # print("EE_POS_TGT: ", EE_POS_TGT)
        # y_std = np.asarray(list(map(float,columns['EpRewSEM'])))
        # print("before clean size mean: ", y_mean.size)
        # print("before clean size std: ", y_std.size)
        # # y_mean = [x for x in y_mean if y_mean is not NaN]
        # y_mean = np.asarray([row for row in y_mean if not np.isnan(row).any()])
        # y_std = np.asarray([row for row in y_std if not np.isnan(row).any()])
        #
        # print("after clean size mean: ", y_mean.size)
        # print("after clean size std: ", y_std.size)

        # x = np.asarray(list(map(float, columns['EVAfter'])))
        x = np.linspace(0, 500, y_mean.size, endpoint=True)

        if smooth is True:
            y_mean = savgol_filter(y_mean, 11, 3)
            # y_std = savgol_filter(y_std, 11, 3)


        print("i: ", i, "; y_mean_max: ", max(y_mean), "; y_mean_min: ", min(y_mean), "; overall mean: ", np.mean(y_mean), "; overall_std: ", y_std)

        # y_upper = y_mean + y_std
        # y_lower = y_mean - y_std
        #
        # # f2 = interp1d(y_upper, y_upper, kind='cubic')
        # plt.fill_between(
        #     x, list(y_lower), list(y_upper), interpolate=True, facecolor=color, linewidth=0.0, alpha=0.3
        # )

        line = plt.plot(x, list(y_mean), color=color, rasterized=False, antialiased=True)

        lines.append(line[0])
        names.append(labels[i])

    plt.legend(lines, names, loc=4)
    # plt.xlim([0,500])
    # plt.ylim([-400,100])
    plt.xlabel("Number of Timesteps")
    plt.ylabel("Reward")

    plt.title(plot_name)
    # plt.xticks([200000, 400000, 600000, 800000, 1000000], ["200K", "400K", "600K", "800K", "1M"])

# env_ids = ["invertedpendulum", "inverteddoublependulum", "reacher", "hopper",\
#             "halfcheetah", "walker2d", "swimmer", "ant"]
plot_names = ["Scara 3DoF", "Scara 3DoF"]
plot_name = "Scara 3DoF"

# plt.figure(figsize=(20,10))
# columns = 4
# i = 0
# for plot_name in plot_names:
datas = []
datas.append("/home/rkojcev/baselines_networks/paper/data/real_robot/ppo1/end_effector_100000000_nsec.csv")
datas.append("/home/rkojcev/baselines_networks/paper/data/real_robot/ppo1/end_effector_10000000_nsec.csv")
datas.append("/home/rkojcev/baselines_networks/paper/data/real_robot/ppo1/end_effector_1000000_nsec.csv")
#
# # datas.append("/home/rkojcev/baselines_networks/paper/data/paper_experiments/deepq/default_hyperpar/progress_max_episode_step_1000.csv")
# datas.append("/home/rkojcev/baselines_networks/paper/data/paper_experiments/final_results/progress_ddpg_g_0_99.csv")
# datas.append("/home/rkojcev/baselines_networks/paper/data/paper_experiments/final_results/progress_deepq_naf_3dof.csv")

#Articulated arm
# datas.append("/tmp/rosrl/GazeboModularArticulatedArm4DOFv1Env/ppo2/progress.csv")

labels = ["100ms", "10ms","1ms"] #"ACKTR",
# labels = [ "DDPG (gamma=0.8)", "DDPG (gamma=0.99)"] #"ACKTR",
plot_results(plot_name, datas, labels, smooth=False)
plt.tight_layout()

plt.savefig('all_rl.png', dpi=400, facecolor='w', edgecolor='w',
        orientation='landscape', papertype='b0', format=None,
        transparent=False, bbox_inches='tight', pad_inches=0.1,
        frameon=None)

plt.show()
