import argparse 
import pathlib 

import logging 
logging.basicConfig(format='%(asctime)s,%(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
log = logging.getLogger("max_hmrc_log")

from mascots.traceAnalysis.RDHistProfiler import RDHistProfiler
from mascots.traceAnalysis.RDHistPlotter import RDHistPlotter

EXCLUSIVE_FLAG = "exclusive"
INCLUSIVE_FLAG = "inclusive"

EXCLUSIVE_MAX_HMRC_DIR = pathlib.Path("/research2/mtc/cp_traces/exclusive_max_hmrc_curve_4k")
INCLUSIVE_MAX_HMRC_DIR =  pathlib.Path("/research2/mtc/cp_traces/inclusive_max_hmrc_curve_4k")
EXCLUSIVE_MAX_HMRC_PLOT_DIR = pathlib.Path("/research2/mtc/cp_traces/exclusive_plot_max_hmrc_curve_4k")
INCLUSIVE_MAX_HMRC_PLOT_DIR = pathlib.Path("/research2/mtc/cp_traces/inclusive_plot_max_hmrc_curve_4k")

EXCLUSIVE_MAX_HMRC_DIR.mkdir(exist_ok=True)
INCLUSIVE_MAX_HMRC_DIR.mkdir(exist_ok=True)
EXCLUSIVE_MAX_HMRC_PLOT_DIR.mkdir(exist_ok=True)
INCLUSIVE_MAX_HMRC_PLOT_DIR.mkdir(exist_ok=True)


def rd_hist_to_max_hmrc(rd_hist_file_path):
    workload_name = rd_hist_file_path.stem 
    max_hmrc_file_name = "{}.csv".format(workload_name)
    plot_file_name = "{}.png".format(workload_name)

    ex_mhrc_output_path = EXCLUSIVE_MAX_HMRC_DIR.joinpath(max_hmrc_file_name)
    in_mhrc_output_path = INCLUSIVE_MAX_HMRC_DIR.joinpath(max_hmrc_file_name)

    if ex_mhrc_output_path.exists():
        print("{} already exists!".format(ex_mhrc_output_path))
        return

    # generate exclusive and inclusive max HMRC and store it in a file 
    profiler = RDHistProfiler(rd_hist_file_path)
    log.info("Loaded {}".format(rd_hist_file_path))
    ex_max_hmrc = profiler.max_hit_miss_ratio(adm_policy=EXCLUSIVE_FLAG)
    in_max_hmrc = profiler.max_hit_miss_ratio(adm_policy=INCLUSIVE_FLAG)

    ex_max_hmrc.tofile(ex_mhrc_output_path, sep="\n")
    in_max_hmrc.tofile(in_mhrc_output_path, sep="\n")
    log.info("Max HMRC file generated for {}".format(rd_hist_file_path))

    # generate exclusive and inclusve plots of max HMRC 
    ex_max_hmrc_plot_path = EXCLUSIVE_MAX_HMRC_PLOT_DIR.joinpath(plot_file_name)
    in_max_hmrc_plot_path = INCLUSIVE_MAX_HMRC_PLOT_DIR.joinpath(plot_file_name)
    plotter = RDHistPlotter(profiler)
    plotter.max_hmrc(ex_max_hmrc_plot_path, adm_policy=EXCLUSIVE_FLAG)
    plotter.max_hmrc(in_max_hmrc_plot_path, adm_policy=INCLUSIVE_FLAG)
    log.info("Max HMRC plot generated for {}".format(rd_hist_file_path))


def max_hmrc_from_folder(rd_histogram_folder):
    for rd_hist_file_path in rd_histogram_folder.iterdir():
        log.info("Processing {}".format(rd_hist_file_path))
        rd_hist_to_max_hmrc(rd_hist_file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Max HMR and save it to a file and plot them.")
    parser.add_argument("rd_histogram_folder", type=pathlib.Path, 
        help="Folder containing RD histogram files")
    args = parser.parse_args()
    max_hmrc_from_folder(args.rd_histogram_folder)