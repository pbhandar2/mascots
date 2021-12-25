import pathlib, json
import numpy as np 
import pandas as pd 
import itertools

class MTGenerator:
    def __init__(self, mt_name, mt_dir):
        """ This class allows users to generate static MT cache configurations from FIO measurements 
            generated using the LocalBench class. 
        """

        self.name = mt_name
        self.mt_dir = pathlib.Path(mt_dir)

        self.min_quantile = 0.05 
        self.max_quantile = 0.95
        self.t1_read_df = None
        self.t1_write_df = None
        self.t2_read_df = None 
        self.t2_write_df = None 
        self.storage_read_df = None 
        self.storage_write_df = None 
        self.device_list = None
        self.device_list_path = pathlib.Path("device_list.csv")
        self.load_device_perf_files()


    def load_device_perf_files(self):
        """ Load the cache device performance CSV inside MT dir.
            The files corresponding to performance data of tier1,
            tier2 and storage are labelled t1, t2 and storage 
            respectively. 
        """

        for device_subdir in self.mt_dir.iterdir():
            if ".csv" in device_subdir.name:
                df = pd.read_csv(device_subdir)
                if "t1" in device_subdir.stem:
                    self.t1_read_df = df[df["op"] == "r"]
                    self.t1_write_df = df[df["op"] == "w"]
                elif "t2" in device_subdir.stem:
                    self.t2_read_df = df[df["op"] == "r"]
                    self.t2_write_df = df[df["op"] == "w"]
                elif "storage" in device_subdir.stem:
                    self.storage_read_df = df[df["op"] == "r"]
                    self.storage_write_df = df[df["op"] == "w"]
                else:
                    raise ValueError

        """ Load the list of device configurations in the repo. 
        """ 
        with self.device_list_path.open("r") as f:
            self.device_list = json.load(f)


    def get_device_json(self, device_entry, device_label):
        """ Get a JSON of a device comprising of performance number from 
            device entry and price and other details from the device list. 
        """
        [device_json] = [d for d in self.device_list if d['label'] == device_label]
        copy_device_json = device_json.copy()
        read_lat = device_entry[0][-1]
        write_lat = device_entry[1][-1]
        copy_device_json["read_lat"] = read_lat
        copy_device_json["write_lat"] = write_lat
        return copy_device_json


    def get_entries(self, quantiles, df):
        """ Get the performance entries of specified quantiles.

        Parameters
        ----------
        quantiles: Series
            series of quantaile and its corresponding value 
        df: DataFrame
            dataframe to search for the quantile entries

        Returns
        -------
        entry_list: list
            list of DataFrame rows corresponding to the quantiles 
        """

        entry_list = []
        for mean_lat in quantiles:
            entry = df.iloc[(df['mean_lat']-mean_lat).abs().argsort()[:1]]
            entry_list.append(entry.iloc[0].to_list())
        return entry_list 


    def generate_devices(self, num_states, tier_index, label):
        """ Generate devices for a specified tier. 

        Parameters
        ----------
        num_states: int
            number of read, write states to capture
        tier_index: int 
            the index of the tier for which to create the device 
        label: str
            the label of the cache device

        Returns
        -------
        device_json_list: list
            list of JSON objects containing cache device information 
        """

        """ Find the value of latency at different quantiles based on the 
            number of states to be used. """

        quantile_array = np.linspace(self.min_quantile, self.max_quantile, num_states)
        if tier_index == 0:
            read_lat_quantiles = self.t1_read_df["mean_lat"].quantile(quantile_array)
            write_lat_quantiles = self.t1_write_df["mean_lat"].quantile(quantile_array)
            read_lat_entry_list = self.get_entries(read_lat_quantiles, self.t1_read_df)
            write_lat_entry_list = self.get_entries(write_lat_quantiles, self.t1_write_df)
        elif tier_index == 1:
            read_lat_quantiles = self.t2_read_df["mean_lat"].quantile(quantile_array)
            write_lat_quantiles = self.t2_write_df["mean_lat"].quantile(quantile_array)
            read_lat_entry_list = self.get_entries(read_lat_quantiles, self.t2_read_df)
            write_lat_entry_list = self.get_entries(write_lat_quantiles, self.t2_write_df)
        elif tier_index == 2:
            read_lat_quantiles = self.storage_read_df["mean_lat"].quantile(quantile_array)
            write_lat_quantiles = self.storage_write_df["mean_lat"].quantile(quantile_array)
            read_lat_entry_list = self.get_entries(read_lat_quantiles, self.t2_read_df)
            write_lat_entry_list = self.get_entries(write_lat_quantiles, self.t2_write_df)
        else:
            raise ValueError

        # Generate a (read, write) combo representing a device 
        lat_entry_list = [read_lat_entry_list, write_lat_entry_list]
        combo_list = list(itertools.product(*lat_entry_list))
        device_json_list = []
        for device_entry in combo_list:
            # convert device_entry to a device object 
            device_json = self.get_device_json(device_entry, label)
            device_json_list.append(device_json)

        return device_json_list 


    def generate_mt_caches(self, t1_label, t2_label, storage_label, output_dir, num_states=3):
        """ Generates MT caches at different system states from FIO output 
            from running LocalBench. 
        """

        t1_devices = self.generate_devices(num_states, 0, t1_label)
        t2_devices = self.generate_devices(num_states, 1, t2_label)
        storage_devices = self.generate_devices(num_states, 2, storage_label)

        mt_device_list = [t1_devices, t2_devices, storage_devices]
        mt_cache_list = list(itertools.product(*mt_device_list))

        for index, mt_cache in enumerate(mt_cache_list):
            mt_cache = list(mt_cache)
            device_file_name = "{}_{}_{}_{}.json".format(
                t1_label, t2_label, storage_label, index)
            device_file_path = pathlib.Path(output_dir).joinpath(device_file_name)

            with device_file_path.open("w+") as f:
                f.write(json.dumps(mt_cache, indent=4))
