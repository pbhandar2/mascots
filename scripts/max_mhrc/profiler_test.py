import pathlib, json 

from mascots.traceAnalysis.RDHistProfiler import RDHistProfiler


RD_HIST_DIR = pathlib.Path("/research2/mtc/cp_traces/rd_hist_4k")
DEVICE_CONFIG_PATH = pathlib.Path("/home/pranav/mtc/mt_config/2/D1_S1_H1.json")

rd_hist_dir = RD_HIST_DIR.joinpath("w106.csv")

mt_config = {}
with open(DEVICE_CONFIG_PATH) as f:
    mt_config = json.load(f)

profiler = RDHistProfiler(rd_hist_dir)
profiler.cost_analysis(mt_config, 5, 256)