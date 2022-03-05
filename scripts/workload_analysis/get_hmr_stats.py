import argparse, json, pathlib 
from scipy import stats
import numpy as np 
from mascots.traceAnalysis.MHMRCProfiler import MHMRCProfiler


class GenerateHMRStats:
    def __init__(self, 
            hmrc_stat_file_path,
            experiment_config="../experiment_config.json"):

        self.hmrc_stat_file_path = hmrc_stat_file_path
        with open(experiment_config, "r") as f:
            self.experiment_config = json.load(f)

        self.rd_hist_dir = pathlib.Path(self.experiment_config["rd_hist_4k_dir"])


    def get_mhmrc(self, workload_name):
        """ Get Max Hit-Miss Ratio Curve
        """

        rd_hist_path = self.rd_hist_dir.joinpath("{}.csv".format(workload_name))
        profiler = MHMRCProfiler(rd_hist_path)  
        return profiler.mhmrc


    def generate_mhmrc_stats(self):
        """ Generate statistics from a Max Hit-Miss Ratio Curve 
            from each workload. 
        """

        for rd_hist_path in self.rd_hist_dir.iterdir():
            print("Processing file {}".format(rd_hist_path))
            workload_name = rd_hist_path.stem 
            mhmrc = self.get_mhmrc(workload_name)
            mhmrc_stats = stats.describe(mhmrc)
            stat_array = [workload_name]
            for stat in mhmrc_stats:
                if type(stat) == tuple:
                    stat_array.append(str(stat[0]))
                    stat_array.append(str(stat[1]))
                else:
                    stat_array.append(str(stat))
            csv_row = ",".join(stat_array)
            print(csv_row)
            with open(self.hmrc_stat_file_path, "a+") as f:
                f.write("{}\n".format(csv_row))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Get HMR stats for all workloads")
    parser.add_argument("--output_path",
        type=pathlib.Path,
        default="/research2/mtc/cp_traces/general/4k_hmrc_stat.csv",
        help="Path to output CSV file")
    args = parser.parse_args()
    
    statGenerator = GenerateHMRStats(args.output_path)
    statGenerator.generate_mhmrc_stats()