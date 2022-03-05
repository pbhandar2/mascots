import numpy as np 
import pathlib 
import matplotlib.pyplot as plt

folder_path = pathlib.Path("/research2/mtc/cp_traces/exclusive_max_hmrc_curve_4k/")

global_array = np.array([0])
max_hmrc = 0 
for i, mhrc_file in enumerate(folder_path.iterdir()):
    cur_array = np.genfromtxt(mhrc_file, delimiter="\n")
    global_array = np.concatenate((cur_array, global_array))
    max_hmrc = max(max_hmrc, max(cur_array))
    print("Loaded: {} {}/106".format(mhrc_file, i))
    print("Length: {}".format(len(global_array)))

    if i > 10: 
        break 

n, bins, patches = plt.hist(global_array, bins=[0, 0.001, 0.002, 0.003, 0.004, 0.005, max_hmrc])

print(n, bins)

plt.xlabel("Max HMR")
plt.ylabel("Probability")
plt.tight_layout()
plt.savefig("hist.png")
plt.close()


