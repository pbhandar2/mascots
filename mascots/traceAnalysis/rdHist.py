import numpy as np
from math import ceil, floor
from collections import Counter 
import pandas as pd 


""" Separates the read and write request from the reuse distance histogram

Args:
    rd_file_path: path to reuse distance file

Returns:
    read and write arrays from a reuse distance file 

"""
def separate_read_write(rd_file_path):
    read_array = []
    write_array = []
    with open(rd_file_path) as f:
        line = f.readline().rstrip()
        while line:
            op_code = line.split(",")[-1]
            reuse_distance = int(line.split(",")[0])

            if op_code == 'r':
                read_array.append(reuse_distance)
            elif op_code == 'w':
                write_array.append(reuse_distance)
            else:
                print("UNKNOWN LABEL FOR READ AND WRITE {}".format(op_code))

            line = f.readline().rstrip()

    return np.array(read_array, dtype=int), np.array(write_array, dtype=int)


""" Creates a RD histogram from an reuse distance array

Args:
    rd_array: the reuse distance array

Returns:
    the reuse distance histogram and the maximum reuse distance

"""
def get_rd_hist(rd_array):

    counter_obj = Counter(rd_array)
    max_reuse_distance = max(rd_array)
    rd_histogram = np.zeros(max_reuse_distance+1, dtype=int)

    # the first entry of histogram is the number of cold misses
    for i in range(max_reuse_distance+1):
        rd_histogram[i] = counter_obj[i-1]

    return rd_histogram, max_reuse_distance


"""    Computes the reuse distance histogram containing read and write count for each bin

Args:
    rd_file_path: path to reuse distance file

Returns:
    reuse distance histogram with read and write count for each bin

"""
def get_rw_hist(rd_file_path):
    read_array, write_array = separate_read_write(rd_file_path)
    read_histogram, max_read_rd = get_rd_hist(read_array)
    write_histogram, max_write_rd = get_rd_hist(write_array)
    max_rd = max(max_read_rd, max_write_rd)
    rw_histogram = np.zeros((max_rd+1, 2))
    rw_histogram[:max_read_rd+1,0] = read_histogram
    rw_histogram[:max_write_rd+1,1] = write_histogram
    return rw_histogram


""" Write np array to a file

Args:
    arr: the array to write
    output_file_path: the path to write it to

"""
def write_np(arr, output_file_path):
    np.savetxt(output_file_path, arr, delimiter=",", fmt='%.d')


""" Computes the histogram from reuse distance and write it to a file

Args:
    rd_file_path: path of the reuse distance file
    rd_hist_file_path: path of the output reuse distance histogram file

"""
def rd_to_rdhist(rd_file_path, rd_hist_file_path):
    rw_hist = get_rw_hist(rd_file_path)
    write_np(rw_hist, rd_hist_file_path)


""" Bin the reuse distance histogram into bin with width specified by bin_range 

Args:
    rdhist: the reuse distance histogram as an np array
    bin_range: the width of each bin 

Returns:
    final_histogram: the final histogram where each bin is of width bin_range
"""
def bin_rdhist(rdhist, bin_range):

    # len(rdhist)-1 because one entry represents cold read and writes not reuse distances 
    num_bins = ceil((len(rdhist)-1)/bin_range)

    # we need to pad the reuse distance histogram with zerso so that we can reshape it 
    padded_rd_hist = np.zeros(shape=(bin_range*num_bins, 2), dtype=int)
    padded_rd_hist[:len(rdhist)-1,:] = rdhist[1:, :]
    bucketed_histogram = padded_rd_hist.reshape(-1, bin_range, 2).sum(axis=1)

    # add that bucketed histogram to the final histogram and append the cold read and write at 0 index
    final_histogram = np.zeros(shape=(num_bins+1, 2), dtype=int)
    final_histogram[0, :] = rdhist[0]
    final_histogram[1:,:] = bucketed_histogram

    return final_histogram


""" Reads the reuse distnace histogram file  

Args:
    rdhist_file: path to file of the reuse distance file

Returns:
    The reuse distnace histogram as an np array 
"""
def read_reuse_hist_file(rdhist_file):
    return np.loadtxt(rdhist_file, delimiter=",")


""" Filter the rd histogram to remove any kind of [0, 0] entries in the tail of the
    histogram. 

Args:
    rdhist: the reuse distance histogram as an np array 

Returns:
    The reuse distance histogram with consequetive [0,0] entries removed from the tail. 
"""
def filter_rdhist(rdhist):
    zero_entry_count = 0
    rdhist_length = len(rdhist)

    # start reading from the end of the histogram 
    for i in range(rdhist_length-1, 1, -1):

        # if both read and write count of the current bin is zero then update counter else break
        if rdhist[i][0] == 0 and rdhist[i][1] == 0:
            zero_entry_count += 1
        else:
            break

    return rdhist[0:rdhist_length-zero_entry_count]


""" Returns the rd histogram from the file 

Args:
    rdhist_file: path to file of the reuse distance file

Returns:
    The reuse distance histogram as np array with consequetive [0,0] entries removed from the tail. 
"""
def get_rdhist_from_file(rdhist_file):
    return filter_rdhist(read_reuse_hist_file(rdhist_file))


""" Bin the reuse distance histogram into bin with width specified by bin_range 

Args:
    rdhist: the reuse distance histogram as an np array
    bin_range: the width of each bin 

Returns:
    final_histogram: the final histogram where each bin is of width bin_range
"""
def bin_rdhist_write_around(rdhist, bin_range):

    print("bin rdhist write around")

    # len(rdhist)-1 because one entry represents cold read and writes not reuse distances 
    num_bins = ceil((len(rdhist)-1)/bin_range)

    # we need to pad the reuse distance histogram with zerso so that we can reshape it 
    padded_rd_hist = np.zeros(bin_range*num_bins, dtype=int)
    padded_rd_hist[:len(rdhist)-1] = rdhist[1:]
    bucketed_histogram = padded_rd_hist.reshape(-1, bin_range).sum(axis=1)

    # add that bucketed histogram to the final histogram and append the cold read and write at 0 index
    final_histogram = np.zeros(num_bins+1, dtype=int)
    final_histogram[0] = rdhist[0]
    final_histogram[1:] = bucketed_histogram

    return final_histogram


""" Filter the rd histogram to remove any kind of [0, 0] entries in the tail of the
    histogram. 

Args:
    rdhist: the reuse distance histogram as an np array 

Returns:
    The reuse distance histogram with consequetive [0,0] entries removed from the tail. 
"""
def filter_rdhist_write_around(rdhist):
    zero_entry_count = 0
    rdhist_length = len(rdhist)

    # start reading from the end of the histogram 
    for i in range(rdhist_length-1, 1, -1):

        # if both read and write count of the current bin is zero then update counter else break
        if rdhist[i] == 0:
            zero_entry_count += 1
        else:
            break

    return rdhist[0:rdhist_length-zero_entry_count]

