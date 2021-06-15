import argparse 
import pathlib 
import numpy as np 
import matplotlib.pyplot as plt 
from mascots.traceAnalysis.rdHist import read_reuse_hist_file, bin_rdhist


RD_HIST_PATH = pathlib.Path("/research2/mtc/cp_traces/rdhist/4k")
RD_HIST_OUTPUT_DIR = pathlib.Path("/research2/mtc/cp_traces/mrc_plot/4k")
PAGE_SIZE_KB = 4
KB_IN_MB=1024


def make_3d_exclusive_mhrc(hrc):
    hrc_3d = np.zeros(shape=(len(hrc), len(hrc)))
    for rd_1 in range(len(hrc)):
        for rd_2 in range(len(hrc)):
            if rd_1 + rd_2 >= len(hrc):
                hrc_3d[rd_1][rd_2] = hrc[-1]
            else:
                hrc_3d[rd_1][rd_2] = hrc[rd_1+rd_2]
    return hrc_3d 


def plot_3d(hrc, output_path, cache_unit_string):
    mhrc = make_3d_exclusive_mhrc(hrc)
    plt.figure(figsize=[14, 10])
    plt.rcParams.update({'font.size': 25})
    ax = plt.subplot(1,1,1, projection='3d')
    (x, y) = np.meshgrid(np.arange(mhrc.shape[0]), np.arange(mhrc.shape[1]))
    surf = ax.plot_surface(x, y, mhrc, cmap='viridis', edgecolor='none')

    ax.set_xlabel("T1 Size ({})".format(cache_unit_string), labelpad=22)
    ax.set_ylabel("T2 Size ({})".format(cache_unit_string), labelpad=22)
    ax.set_zlabel("Hit Rate", labelpad=18)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def get_hrc(rd_hist):
    total_req = np.sum(rd_hist)
    print("Total Req: {}".format(total_req))
    cumsum_rd_hist = np.zeros(len(rd_hist))
    cumsum_rd_hist[1:] = np.sum(np.cumsum(rd_hist[1:, :], axis=0), axis=1)
    return np.divide(cumsum_rd_hist, total_req)


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
    

def main(rdhist_path, bin_size, output_path_hrc, output_path_rdhist, output_path_mhrc):
    print("Plotting HRC: {} with binsize {} and output {}".format(rdhist_path, bin_size, output_path_hrc))
    rd_hist = bin_rdhist(read_reuse_hist_file(rdhist_path), bin_size)
    hrc = get_hrc(rd_hist)

    if bin_size < int(KB_IN_MB/PAGE_SIZE_KB):
        cache_unit_string = "{}KB".format(PAGE_SIZE_KB*bin_size)
    elif bin_size == 256:
        cache_unit_string = "MB"
    else:
        cache_unit_string = "{}MB".format(int(bin_size/int(KB_IN_MB/PAGE_SIZE_KB)))

    # for hrc_zoom in range(10, 101, 10):
    #     print("Plotting HRC with zoom level {}", hrc_zoom)
    #     plt.figure(figsize=[14, 10])
    #     plt.rcParams.update({'font.size': 35})
    #     ax = plt.subplot(1,1,1)
    #     ax.plot(hrc[:int(len(hrc)*hrc_zoom/100)], "--", linewidth=3)
    #     ax.set_ylim(0, 1.01)

    #     plt.xlabel("Cache Size ({})".format(cache_unit_string))
    #     plt.ylabel("Hit Rate")
    #     plt.tight_layout()
    #     plt.savefig(output_path_hrc.joinpath("{}_{}_{}.png".format(rdhist_path.stem, cache_unit_string, hrc_zoom)))
    #     plt.close()


    # print("RDHist path: {}, bin size: {}, output: {}".format(rdhist_path, bin_size, output_path_rdhist))
    # read_hist = rd_hist[:,0]
    # write_hist = rd_hist[:,1]
    # plt.figure(figsize=[14, 10])
    # plt.rcParams.update({'font.size': 35})
    # ax = plt.subplot(1,1,1)
    # plot_hist_main(ax, rd_hist[1:,:])
    # plt.xlabel("Reuse Distance ({})".format(cache_unit_string))
    # plt.ylabel("log(Hit Count)")
            
    # # Base 10 
    # plt.ylabel("Count")
    # plt.tight_layout()
    # plt.savefig(output_path_rdhist.joinpath("{}_{}_base.png".format(rdhist_path.stem, cache_unit_string)))

    # # log scale
    # ax.set_yscale("log")
    # plt.ylabel("log(Count)")
    # plt.tight_layout()
    # plt.savefig(output_path_rdhist.joinpath("{}_{}_log.png".format(rdhist_path.stem, cache_unit_string)))

    # plt.close()


    print("MHRC --> RDHist path: {}, bin size: {}, output: {}".format(rdhist_path, bin_size, output_path_mhrc))
    mhrc_output_path = output_path_mhrc.joinpath("{}_{}.png".format(rdhist_path.stem, cache_unit_string))
    plot_3d(hrc, mhrc_output_path, cache_unit_string)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot RD histogram")
    parser.add_argument("rd_hist_filename", help="File name of RD histogram")
    parser.add_argument("--b", default=2560, type=int, help="The bin size. Default: 2560 (equals 10MB in 4KB page)")
    parser.add_argument("--o_hrc", default=pathlib.Path.cwd().joinpath("hrc"), type=pathlib.Path, help="The output path of the plot")
    parser.add_argument("--o_rdhist", default=pathlib.Path.cwd().joinpath("rdhist_4k"), type=pathlib.Path, help="The output path of the plot")
    parser.add_argument("--o_3dhrc", default=pathlib.Path.cwd().joinpath("3d_hrc"), type=pathlib.Path, help="The output path of the plot")
    args = parser.parse_args()
    main(RD_HIST_PATH.joinpath(args.rd_hist_filename), args.b, args.o_hrc, args.o_rdhist, args.o_3dhrc)