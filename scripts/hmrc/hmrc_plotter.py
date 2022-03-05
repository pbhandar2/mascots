import json, sys, pathlib
from HMRC import HMRC

experiment_config_file_path = "../experiment_config.json"
with open(experiment_config_file_path, "r") as f:
    experiment_config = json.load(f) 

workload_path = pathlib.Path(experiment_config["rd_hist_4k_dir"]).joinpath("{}.csv".format(sys.argv[1]))

hmrc = HMRC(workload_path)
hmrc.plot_hmrc("D1_S1_H1", 12)
hmrc.plot_hmrc("D1_S1_H1", 23)