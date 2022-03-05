import sys 

from mascots.traceAnalysis.RDHistProfiler import RDHistProfiler
from mascots.traceAnalysis.RDHistPlotter import RDHistPlotter

if __name__ == "__main__":
    profiler = RDHistProfiler(sys.argv[1])
    plotter = RDHistPlotter(profiler)
    # profiler.plot_mrc(sys.argv[2])
    plotter.hrc(sys.argv[2])