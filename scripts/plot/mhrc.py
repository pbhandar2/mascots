import sys 
import pathlib 

from mascots.traceAnalysis.RDHistProfiler import RDHistProfiler

if __name__ == "__main__":
    rd_hist_file_path = pathlib.Path(sys.argv[1])
    profiler = RDHistProfiler(rd_hist_file_path)
    workload_name = rd_hist_file_path.stem 

    # for filter_size_mb in range(0,20000, 5000):
    #     profiler.plot_mhrc(output_path="./w81-w85/{}_{}.png".format(workload_name, filter_size_mb), 
    #     filter_size_mb=filter_size_mb, y_log_scale=True)

    profiler.plot_hmrc(output_path="./w81-w85/log_{}_{}.png".format(workload_name, "00"), 
        filter_size_mb_array=range(20000, 0, -5000), y_log_scale=True)

    profiler.plot_hmrc(output_path="./w81-w85/{}_{}.png".format(workload_name, "00"), 
        filter_size_mb_array=range(20000, 0, -5000))