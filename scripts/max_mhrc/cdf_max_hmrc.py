import numpy as np 
import pathlib 
import matplotlib.pyplot as plt

max_hmrc = 0 
bins = np.arange(0.01, 1.5, 0.01)
bins = np.concatenate((bins, np.array([np.inf])))

global_array = np.array([0])
folder_path = pathlib.Path("/research2/mtc/cp_traces/exclusive_max_hmrc_curve_4k/")
for i, mhrc_file in enumerate(folder_path.iterdir()):
    workload_number = int(mhrc_file.stem.split("w")[1])
    cur_array = np.genfromtxt(mhrc_file, delimiter="\n")
    global_array = np.concatenate((cur_array, global_array))
    max_hmrc = max(max_hmrc, max(cur_array))
    print("Workload loaded: w{}, {}/106, Max HMR: {}".format(workload_number, i, max_hmrc))

hist, bins, patches = plt.hist(global_array, 
    bins, 
    density=True, 
    cumulative=True, 
    histtype='step', 
    color='purple')

plt.xticks(np.arange(0, 1.5, 0.2, dtype=float))
plt.xlabel("Max HMR")
plt.ylabel("Probability")
plt.tight_layout()
plt.savefig("hist.png")
plt.close()