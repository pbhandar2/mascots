import argparse 
import pathlib 
import matplotlib.pyplot as plt 
from mascots.traceAnalysis.rdHist import read_reuse_hist_file, bin_rdhist

RD_HIST_PATH = pathlib.Path("/research2/mtc/cp_traces/rdhist/4k")
RD_HIST_OUTPUT_DIR = pathlib.Path("/research2/mtc/cp_traces/rdhist_plot/4k")


def plot_hist_main(ax, rd_hist):
    read_hist = []
    write_hist = []
    for _bin in rd_hist: 
        read_hist.append(_bin[0])
        write_hist.append(_bin[1])
    width = 1
    ind = range(0, len(rd_hist))
    ax.bar(ind, write_hist, width, color="red", label="Write")
    ax.bar(ind, read_hist, width, color="blue", label="Read", bottom=write_hist)
    ax.legend()
    ax.set_yscale("log")


def main(rdhist_path, bin_size, output_path):
    print("RDHist path: {}, bin size: {}, output: {}".format(rdhist_path, bin_size, output_path))
    rd_hist = bin_rdhist(read_reuse_hist_file(rdhist_path), bin_size)
    read_hist = rd_hist[:,0]
    write_hist = rd_hist[:,1]

    plt.figure(figsize=[14, 10])
    plt.rcParams.update({'font.size': 35})
    ax = plt.subplot(1,1,1)
    plot_hist_main(ax, rd_hist[1:,:])
    plt.xlabel("Reuse Distance ({}MB)".format(int(bin_size/(256))))
    plt.ylabel("log(Hit Count)")
    plt.tight_layout()
    plt.savefig(output_path)





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot RD histogram")
    parser.add_argument("rd_hist_filename", help="File name of RD histogram")
    parser.add_argument("--b", default=2560, type=int, help="The bin size. Default: 2560 (equals 10MB in 4KB page)")
    parser.add_argument("--o", default=pathlib.Path.cwd().joinpath("rdhist_4k"), type=pathlib.Path, help="The output path of the plot")
    args = parser.parse_args()

    main(RD_HIST_PATH.joinpath(args.rd_hist_filename), args.b, args.o)