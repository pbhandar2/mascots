import json, pathlib 
import numpy as np 
import matplotlib.pyplot as plt

from mascots.traceAnalysis.MTConfigLib import MTConfigLib 
from mascots.traceAnalysis.RDHistProfiler import RDHistProfiler

class HMRC:
    def __init__(self, rd_hist_path, page_size=4096, experiment_config_file_path="../experiment_config.json"):

        self.rd_hist_path = pathlib.Path(rd_hist_path)
        self.workload_name = self.rd_hist_path.stem 
        self.rd_hist_profiler = RDHistProfiler(rd_hist_path, page_size=page_size)
        self.mt_lib = MTConfigLib()

        self.experiment_config_file_path = experiment_config_file_path
        with open(self.experiment_config_file_path, "r") as f:
            self.experiment_config = json.load(f) 
    

    def get_hmrc(self, mt_config, cost):
        max_t1_size = self.mt_lib.get_max_tier_size(mt_config, 0, 
                            self.rd_hist_profiler.allocation_unit, cost)

        hmrc_array = np.zeros(max_t1_size, dtype=float)
        for t1_size in range(1, max_t1_size):
            t2_size = self.mt_lib.get_max_t2_size(mt_config, t1_size, cost,
                                                    self.rd_hist_profiler.allocation_unit)
            hit_array = self.rd_hist_profiler.get_exclusive_cache_hits([t1_size, t2_size])
            hit = hit_array[1][0]
            miss = hit_array[1][1] + hit_array[-1][0] + hit_array[-1][1]
            hmrc_array[t1_size-1] = hit/miss 
        return hmrc_array

    
    def plot_hmrc(self, mt_config_label, cost):
        device_config_dir = pathlib.Path(self.experiment_config["device_config_dir"])
        mt_config_label_path = device_config_dir.joinpath("2", "{}.json".format(mt_config_label))
        with open(mt_config_label_path) as f:
            mt_config = json.load(f)
        
        hmrc = self.get_hmrc(mt_config, cost)
        fig, ax = plt.subplots(figsize=(14,7))
        ax.plot(hmrc)
        plt.savefig("hmrc_plots/{}_{}.png".format(mt_config_label, cost))
        plt.close()