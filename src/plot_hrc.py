import argparse 
import pathlib 
import numpy as np 
import matplotlib.pyplot as plt 
from mascots.traceAnalysis.rdHist import read_reuse_hist_file, bin_rdhist


RD_HIST_PATH = pathlib.Path("/research2/mtc/cp_traces/rdhist/4k")
RD_HIST_OUTPUT_DIR = pathlib.Path("/research2/mtc/cp_traces/mrc_plot/4k")
PAGE_SIZE_KB = 4
KB_IN_MB=1024

def get_hrc(rd_hist):
    total_req = np.sum(rd_hist)
    cumsum_rd_hist = np.zeros(len(rd_hist))
    cumsum_rd_hist[1:] = np.sum(np.cumsum(rd_hist[1:, :], axis=0), axis=1)
    return np.divide(cumsum_rd_hist, total_req)


def main(rdhist_path, bin_size, output_path):
    print("Plotting HRC: {} with binsize {} and output {}".format(rdhist_path, bin_size, output_path))
    rd_hist = bin_rdhist(read_reuse_hist_file(rdhist_path), bin_size)
    hrc = get_hrc(rd_hist)
    plt.figure(figsize=[14, 10])
    plt.rcParams.update({'font.size': 35})
    ax = plt.subplot(1,1,1)
    ax.plot(hrc, "--", linewidth=3)
    ax.set_ylim(0, 1.02)

    if bin_size < int(KB_IN_MB/PAGE_SIZE_KB):
        cache_unit_string = "{}KB".format(PAGE_SIZE_KB*bin_size)
    elif bin_size == 256:
        cache_unit_string = "MB"
    else:
        cache_unit_string = "{}MB".format(cache_unit)

    plt.xlabel("Cache Size ({})".format(cache_unit_string))
    plt.ylabel("Hit Rate")
    plt.tight_layout()
    plt.savefig(output_path.joinpath("{}.png".format(rdhist_path.stem)))
    plt.close()

    read_hist = rd_hist[:,0]
    write_hist = rd_hist[:,1]

    print("RDHist path: {}, bin size: {}, output: {}".format(rdhist_path, bin_size, output_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot RD histogram")
    parser.add_argument("rd_hist_filename", help="File name of RD histogram")
    parser.add_argument("--b", default=2560, type=int, help="The bin size. Default: 2560 (equals 10MB in 4KB page)")
    parser.add_argument("--o", default=pathlib.Path.cwd().joinpath("hrc"), type=pathlib.Path, help="The output path of the plot")
    args = parser.parse_args()
    main(RD_HIST_PATH.joinpath(args.rd_hist_filename), args.b, args.o)