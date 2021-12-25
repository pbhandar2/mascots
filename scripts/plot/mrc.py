import sys 

from mascots.traceAnalysis.RDHistProfiler import RDHistProfiler

if __name__ == "__main__":
    profiler = RDHistProfiler(sys.argv[1])
    profiler.plot_mrc(sys.argv[2])

