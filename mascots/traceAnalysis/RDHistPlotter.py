import math 
import numpy as np 
import matplotlib.pyplot as plt

class RDHistPlotter:
    """ The class plots the following plots:
        - Miss Rate Curve (MRC)
        - Hit Rate Curve (HRC)
        - Miss-Hit Ratio Curve (MHRC)
        - Min Miss-Hit Ratio Curve (M-MHRC)

    Attributes
    ----------
    profiler : RDHistProfiler
        the RDHistProfiler where the data is pulled to plot 
    x_axis_step_sizes : np.array
        np array containing the possible x-axis step sizes in MB 
    """

    def __init__(self, profiler):
        """
        Parameters
        ----------
        profiler : RDHistProfiler
            the RDHistProfiler where the data is pulled to plot 
        """

        self.profiler = profiler 
        self.x_axis_step_sizes = np.array([10, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000], dtype=int)
        self.symbols = ["d", "8", "X", ">"]
        self.num_xticks = 10 
    
    
    def setup_xaxis(self,
        ax,
        min_cache_size_mb,
        max_cache_size_mb):
        """ Setup the x-axis of the plot. 

        Parameters
        ----------
        ax : matplotlib axes
            setup the x-axis of the given matplotlib axes 
        min_cache_size_mb: int
            min cache size in MB
        max_cache_size_mb: int
            max cache size in MB 
        """
        
        cache_size_range_mb = max_cache_size_mb - min_cache_size_mb
        idx = (np.abs(self.x_axis_step_sizes - int(cache_size_range_mb//self.num_xticks))).argmin()
        xtick_stepsize_mb = self.x_axis_step_sizes[idx]

        # based on the step size for x-ticks pick the correct unit MB/GB 
        label_unit = "GB" if xtick_stepsize_mb >= 100 else "MB"
        xtick_label_array, xtick_array = [], []
        for cache_size_mb in range(min_cache_size_mb, max_cache_size_mb+1):
            if cache_size_mb % xtick_stepsize_mb == 0:
                if label_unit == "GB":
                    xtick_label_array.append("{}".format(cache_size_mb/1e3))
                else:
                    xtick_label_array.append("{}".format(cache_size_mb))
                xtick_array.append(np.floor(cache_size_mb*1e6/self.profiler.page_size).astype(int))

        ax.set_xticks(xtick_array)
        ax.set_xticklabels(xtick_label_array)
        ax.set_xlabel("Cache Size ({})".format(label_unit))


    def hrc(self,
        output_path,
        min_cache_size_mb=0,
        max_cache_size_mb=-1,
        yscale_log_flag=False):
        """ Plot MRC based on the histogram of the profiler.

        Parameters
        ----------
        output_path: str
            output path of the plot 
        min_cache_size_mb: int
            min cache size in MB (Default: 0) (optional) 
        max_cache_size_mb: int
            max cache size in MB (Default: -1) (optional) 
        num_xticks: int
            number of xticks (Default: 10) (optional) 
        """
        
        fig, ax = plt.subplots(figsize=(14,7))
        hrc = self.profiler.hrc 

        start_index = int((min_cache_size_mb*1e6)//self.profiler.page_size)
        if max_cache_size_mb == -1:
            end_index = len(hrc) - 1
            max_cache_size_mb = math.ceil(len(hrc)*self.profiler.page_size/1e6)
        else:
            end_index = int((max_cache_size_mb*1e6)//self.profiler.page_size)

        ax.plot(hrc[start_index:end_index+1], 
            marker=self.symbols[0], 
            markersize=5,
            alpha=0.8,
            markevery=int(len(hrc)/100))
        self.setup_xaxis(ax, min_cache_size_mb, max_cache_size_mb)

        if yscale_log_flag:
            ax.set_yscale("log")
            ax.set_ylabel("log(Hit Rate)")
        else:
            ax.set_ylabel("Hit Rate")

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()


    def multi_hrc(self,
        output_path,
        min_cache_size_mb=0,
        max_cache_size_mb=-1,
        yscale_log_flag=False):
        """ Plot MRC based on the histogram of the profiler.

        Parameters
        ----------
        output_path: str
            output path of the plot 
        min_cache_size_mb: int
            min cache size in MB (Default: 0) (optional) 
        max_cache_size_mb: int
            max cache size in MB (Default: -1) (optional) 
        num_xticks: int
            number of xticks (Default: 10) (optional) 
        """
        
        fig, ax = plt.subplots(figsize=(14,7))

        read_hrc = self.profiler.rd_hist[:, 0].cumsum()/self.profiler.read_count 
        write_hrc = self.profiler.rd_hist[:, 1].cumsum()/self.profiler.write_count 

        start_index = int((min_cache_size_mb*1e6)//self.profiler.page_size)
        if max_cache_size_mb == -1:
            end_index = len(read_hrc) - 1
            max_cache_size_mb = math.ceil(len(read_hrc))
        else:
            end_index = int((max_cache_size_mb*1e6)//self.profiler.page_size)

        print(len(read_hrc))
        print(start_index, end_index)

        ax.plot(read_hrc[start_index:end_index+1], 
            marker=self.symbols[0], 
            markersize=8,
            alpha=0.8,
            markevery=int(len(read_hrc)/100),
            label="Read")
        ax.plot(write_hrc[start_index:end_index+1], 
            marker=self.symbols[1], 
            markersize=8,
            alpha=0.8,
            markevery=int(len(write_hrc)/100),
            label="Write")
        self.setup_xaxis(ax, min_cache_size_mb, max_cache_size_mb)

        if yscale_log_flag:
            ax.set_yscale("log")
            ax.set_ylabel("log(Hit Rate)")
        else:
            ax.set_ylabel("Hit Rate")

        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.set_xlabel("Cache Size (MB)", fontsize=15)
        ax.set_ylabel("Hit Rate", fontsize=15)

        plt.legend(fontsize=15)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()


    def max_hmrc(self,
        output_path,
        min_cache_size_mb=0,
        max_cache_size_mb=-1,
        adm_policy="exclusive"):
        """ Plot max HMRC based on the histogram of the profiler.

        Parameters
        ----------
        output_path: str
            output path of the plot 
        min_cache_size_mb: int
            min cache size in MB (Default: 0) (optional) 
        max_cache_size_mb: int
            max cache size in MB (Default: -1) (optional) 
        adm_policy : string 
            "exclusive" or "inclusive" admission policy 
        """
        fig, ax = plt.subplots(figsize=(14,7))

        max_hmrc_array = self.profiler.max_hit_miss_ratio(adm_policy=adm_policy)

        start_index = int((min_cache_size_mb*1e6)//self.profiler.page_size)
        if max_cache_size_mb == -1:
            end_index = len(max_hmrc_array) - 1
            max_cache_size_mb = math.ceil(len(max_hmrc_array)*self.profiler.page_size/1e6)
        else:
            end_index = int((max_cache_size_mb*1e6)//self.profiler.page_size)

        ax.plot(max_hmrc_array[start_index:end_index],
            marker=self.symbols[0], 
            markersize=5,
            alpha=0.8,
            markevery=int(len(max_hmrc_array)/100))
        self.setup_xaxis(ax, min_cache_size_mb, max_cache_size_mb)

        ax.set_ylabel("Max HMR (Hit-Miss Ratio)")

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

        return max_hmrc_array