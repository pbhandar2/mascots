import argparse, pathlib, json, math 
from urllib.parse import MAX_CACHE_SIZE 
import numpy as np 
import matplotlib.pyplot as plt

from mascots.traceAnalysis.MHMRCProfiler import MHMRCProfiler
from mascots.traceAnalysis.MTConfigLib import MTConfigLib


class PlotScaledHMRC:
    """ Plot the scaled HMRC of a given workload
    """
    def __init__(self, workload_name, output_dir):
        """
        Parameters
        ----------
        output_dir : pathlib.Path
            the directory to output file 
        """

        self.output_dir = output_dir
        self.workload_name = workload_name 
        self.rd_hist_dir = pathlib.Path("/research2/mtc/cp_traces/rd_hist_4k/")
        self.mt_config_dir = pathlib.Path("/home/pranav/mtc/mt_config/2")
        self.rd_hist_path = self.rd_hist_dir.joinpath("{}.csv".format(workload_name))
        self.allocation_unit = 256 
        self.mhmrc_profiler = MHMRCProfiler(self.rd_hist_path)
        self.mt_config_lib = MTConfigLib()


    def get_mt_cache(self, mt_label):
        """ Get the list of dictionaries using MT label 
        """

        mt_config_file_path = self.mt_config_dir.joinpath("{}.json".format(mt_label))
        with open(mt_config_file_path) as f:
            mt_config = json.load(f)
        return mt_config 


    def get_scaled_mhmrc(self, mt_label, cost):
        """ Get Scaled Max Hit-Miss Ratio Curve 

        Parameter
        ---------
        mt_label : str 
            the label of MT cache 
        cost : int 
            the cost of cache in dollar 

        Return 
        ------
        scaled_mhmrc : np.array 
            the array containing the scaled Max Hit-Miss Ratio 
        """

        mt_config = self.get_mt_cache(mt_label)
        scaled_mhmrc = self.mhmrc_profiler.get_scaled_mhmrc(self.mhmrc_profiler.mhmrc,
            mt_config,
            cost)
        return scaled_mhmrc

    
    def plot_scaled_mhmrc(self, scaled_mhmrc, output_path):
        """ Plot the scaled MHMRC 

        Parameter
        ---------
        scaled_mhmrc : np.array
            the scaled Max Hit-Miss Ratio Curve 
        output_path : str 
            the output path 
        """

        fig, ax = plt.subplots(figsize=(14,7))

        max_value_index = np.argmax(scaled_mhmrc)
        ax.scatter(max_value_index, 
            scaled_mhmrc[max_value_index], 
            s=250,
            marker="*",
            color='r',
            label="Max Value at {}MB".format(max_value_index+1))
        ax.plot(range(1, len(scaled_mhmrc)+1), scaled_mhmrc, "-^",
            markersize=10,
            alpha=0.75,
            markevery=int(len(scaled_mhmrc)/100))

        plt.xlabel("Tier 1 Size (MB)")
        plt.ylabel("Scaled Hit-Miss Ratio Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()


    def plot_all_scaled_mhmrc(self, mt_label):
        """ Plot scaled HMRC for all cost values 
        """

        mt_config = self.get_mt_cache(mt_label)
        max_cache_size = self.mhmrc_profiler.rd_hist_profiler.max_cache_size
        max_cost = self.mt_config_lib.get_max_cost(mt_config, max_cache_size, 
            self.mhmrc_profiler.rd_hist_profiler.allocation_unit)

        for cost in range(1, max_cost+2):
            scaled_mhmrc = self.get_scaled_mhmrc(mt_label, cost)
            output_path = self.output_dir.joinpath("{}-{}.png".format(cost, mt_label))
            self.plot_scaled_mhmrc(scaled_mhmrc, output_path)


    def plot_multiple_scaled_mhmrc(self, mt_label, num_lines=5):
        """ Plot multiple sclaed HMRC in a single plot 
        """

        mt_config = self.get_mt_cache(mt_label)
        max_cache_size = self.mhmrc_profiler.rd_hist_profiler.max_cache_size
        max_cost = self.mt_config_lib.get_max_cost(mt_config, max_cache_size, 
            self.mhmrc_profiler.rd_hist_profiler.allocation_unit)

        symbol_array = ["-*", "-^", "-s", "-X", "-D", "-8", "-P"]
        fig, ax = plt.subplots(figsize=(14,7))
        for index, cost_ratio in enumerate([0.05, 0.4, 0.75]):
            cost = math.floor(cost_ratio*max_cost)
            scaled_mhmrc = self.get_scaled_mhmrc(mt_label, cost)
            ax.plot(range(1, len(scaled_mhmrc)+1), scaled_mhmrc, 
                symbol_array[index],
                markersize=12,
                alpha=0.75,
                markevery=int(len(scaled_mhmrc)/100),
                label="${}".format(cost))

        output_path = self.output_dir.joinpath("0-{}.pdf".format(mt_label))

        xticks = range(0, len(scaled_mhmrc), 2000)
        xtick_labels = [str(x/1000) for x in xticks]

        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels)
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.set_xlabel("Tier 1 Size (GB)", fontsize=25)
        ax.set_ylabel("Hit-Miss Ratio", fontsize=25)

        plt.legend(fontsize=25, markerscale=1.5)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

        print("Plotted {}".format(output_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot scaled HMRC")
    parser.add_argument("workload_name", 
        help="The name of the workload")
    parser.add_argument("mt_label", 
        help="The label of MT configuration")
    parser.add_argument("--output_dir", 
        default=pathlib.Path("/research2/mtc/cp_traces/scaled_hmrc_plot"),
        type=pathlib.Path,
        help="Directory to output plots")
    args = parser.parse_args()

    workload_output_dir = args.output_dir.joinpath(args.workload_name)
    workload_output_dir.mkdir(exist_ok=True, parents=True)
    plotter = PlotScaledHMRC(args.workload_name, workload_output_dir)
    #plotter.plot_all_scaled_mhmrc(args.mt_label)
    plotter.plot_multiple_scaled_mhmrc(args.mt_label)