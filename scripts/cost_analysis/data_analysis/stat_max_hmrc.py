from collections import defaultdict
import pandas as pd 
import json, pathlib, math 
import numpy as np 
import matplotlib.pyplot as plt
from tabulate import tabulate

from mascots.traceAnalysis.MHMRCProfiler import MHMRC_ROW_JSON
from mascots.traceAnalysis.RDHistProfiler import OPT_ROW_JSON 

class StatMaxHMRC:
    def __init__(self,
        mt_label,
        algorithm,
        step_size,
        experiment_config="../../experiment_config.json"):

        with open(experiment_config, "r") as f:
            self.experiment_config = json.load(f)

        self.mt_label = mt_label
        self.algorithm = algorithm
        self.step_size = step_size

        self.hmrc_dir = pathlib.Path(self.experiment_config["mhmrc_cost_analysis_dir"])
        self.exhaustive_dir = pathlib.Path(self.experiment_config["ex_cost_analysis_dir"])
        self.per_workload_points_vs_error_plot_dir = \
            pathlib.Path(self.experiment_config["per_workload_points_vs_error_plot_dir"])
        
        self.wb_ex_data, self.wt_ex_data = {}, {}
        self.wb_hmrc_data, self.wt_hmrc_data = {}, {}
        self.wb_error_data, self.wt_error_data = {}, {}
        self.markerlist = ["D", "X", "s", "P", "O"]


    def load_data(self, workload_name, write_policy):
        """ Load data based on write policy and workload name 

        Parameters
        ----------
        write_policy : str 
            the write policy "wb" write-back or "wt" write-through 
        workload_name : str 
            the name of the workload 
        """
        
        data_dir = self.hmrc_dir.joinpath(self.mt_label, 
                                            write_policy, 
                                            workload_name,
                                            self.algorithm, 
                                            str(self.step_size))
        
        workload_data = {}
        for data_file_path in data_dir.iterdir():
            file_eval_count = int(data_file_path.stem)

            df = pd.read_csv(data_file_path, names=MHMRC_ROW_JSON)
            df.astype({
                "cost": int, 
                "st_latency": float, 
                "latency": float
            })
            workload_data[file_eval_count] = df 
    
            if write_policy == "wb":
                self.wb_hmrc_data[workload_name] = workload_data
            elif write_policy == "wt":
                self.wt_hmrc_data[workload_name] = workload_data
            else:
                raise ValueError("write-policy provided not supported.")
        
        df = pd.read_csv(self.exhaustive_dir.joinpath(self.mt_label, 
                                                        write_policy, 
                                                        "{}.csv".format(workload_name)),
                            names=OPT_ROW_JSON)
        df.astype({
            "cost": int, 
            "st_latency": float, 
            "mt_p_latency": float,
            "mt_np_latency": float
        })

        if write_policy == "wb":
            self.wb_ex_data[workload_name] = df
            self.wb_error_data[workload_name] = {}
        elif write_policy == "wt":
            self.wt_ex_data[workload_name] = df
            self.wt_error_data[workload_name] = {}
        else:
            raise ValueError("write-policy provided not supported.")


    def get_error(self, workload_name, write_policy, num_points_array=[0,1,5,10,50,100]):
        """ Get error for a given workload and write-policy. 

            Parameters
            ----------
            workload_name : str
                the label of the workload 
            write_policy : str 
                "wb" for write-back and "wt" for write-through 
            
            Return
            ------
            error : dict 
                dict[*cost*][*num_points*] = [*opt_type*, *percent_eval*, *percent_erro*]
        """

        if write_policy == "wb":
            ex_df = self.wb_ex_data[workload_name]
            hmrc_data = self.wb_hmrc_data[workload_name]
        else:
            ex_df = self.wt_ex_data[workload_name]
            hmrc_data = self.wt_hmrc_data[workload_name]
        
        error = {}
        for _, ex_row in ex_df.iterrows():
            cost_value = ex_row["cost"]
            error_output = {}
            for num_points in hmrc_data:
                
                if num_points not in num_points_array:
                    continue 

                hmrc_df = hmrc_data[num_points]
                hmrc_row = hmrc_df[hmrc_df["cost"]==cost_value]

                if len(hmrc_row) > 0:
                    ex_latency = ex_row[["st_latency", "mt_p_latency", "mt_np_latency"]].min(axis=0)
                    hmrc_latency = hmrc_row.iloc[0]["latency"]
                    if hmrc_row.iloc[0]["t2_size"] > 0:
                        opt_type = 1
                    else:
                        opt_type = 0 

                    percent_error = 100*(hmrc_latency - ex_latency)/ex_latency
                    percent_eval = 100*hmrc_row.iloc[0]["num_eval"]/hmrc_row.iloc[0]["max_points"]
                    error_output[num_points] = [opt_type, percent_eval, percent_error]
                    
            error[cost_value] = error_output
        return error


    def print_predicted_vs_opt(self, write_policy, workload_name, cost_value):
        # get the data based on write-policy and workload name 
        if write_policy == "wb":
            print(self.wb_data.keys())
            max_cost = self.wb_data[workload_name][0]["cost"].max()
            data = self.wb_data[workload_name]
        elif write_policy == "wt":
            max_cost = self.wt_data[workload_name][0]["cost"].max()
            data = self.wt_data[workload_name]
        else:
            raise ValueError("write-policy provided not supported.")

        exhaustive_file_path = self.exhaustive_dir.joinpath(self.mt_label,
                                    write_policy, 
                                    "{}.csv".format(workload_name))
        exhaustive_df = pd.read_csv(exhaustive_file_path, names=OPT_ROW_JSON)
        exhaustive_df.astype({
            "st_latency": float, 
            "mt_p_latency": float, 
            "mt_np_latency": float
        })

        out = []
        for key in data:

            mhmrc_df = data[key][data[key]["cost"] == cost_value]
            ex_df = exhaustive_df[exhaustive_df["cost"] == cost_value]

            try:
                assert(len(mhmrc_df)<=1 and len(ex_df) <= 1)
            except AssertionError:
                print(mhmrc_df)
                print(ex_df)
                raise AssertionError

            if len(mhmrc_df) > 0 and len(ex_df) > 0:
                mhmrc_row = mhmrc_df.iloc[0]
                ex_row = ex_df.iloc[0]
                ex_latency = ex_row[["st_latency", "mt_p_latency", "mt_np_latency"]].min(axis=0)
                if ex_row["mt_opt_flag"] == 0:
                    ex_size_array = "-".join([str(ex_row["st_size"]), "0"])
                elif ex_row["mt_opt_flag"] == 1:
                    ex_size_array = ex_row["mt_p_size_array"]
                else:
                    ex_size_array = ex_row["mt_np_size_array"]
                mhmrc_latency = mhmrc_row["latency"]
                error = mhmrc_latency - ex_latency
                percent_error = 100*error/ex_latency
                mhmrc_t1 = mhmrc_row["t1_size"]
                mhmrc_t2 = mhmrc_row["t2_size"]
                assert(mhmrc_t1>=0 and mhmrc_t2>=0)
                out.append([key, mhmrc_t1, mhmrc_t2, ex_size_array, percent_error])

        out = sorted(out, key=lambda x: x[0])
        print(tabulate(out[:500], headers=["n", "mhmrc_t1", "mhmrc_t2", "opt", "error%"]))
            

    def plot_num_points_vs_error(self, write_policy, workload_name):
        """ Plot number of points vs error for a given workload 

        Parameters
        ----------
        write_policy : str 
            the write policy "wb" write-back or "wt" write-through 
        workload_name : str 
            the name of the workload 
        """

        # get the data based on write-policy and workload name 
        if write_policy == "wb":
            max_cost = self.wb_data[workload_name][0]["cost"].max()
            data = self.wb_data[workload_name]
        elif write_policy == "wt":
            print(self.wt_data.keys())
            max_cost = self.wt_data[workload_name][0]["cost"].max()
            data = self.wt_data[workload_name]
        else:
            raise ValueError("write-policy provided not supported.")

        exhaustive_file_path = self.exhaustive_dir.joinpath(self.mt_label,
                                    write_policy, 
                                    "{}.csv".format(workload_name))
        exhaustive_df = pd.read_csv(exhaustive_file_path, names=OPT_ROW_JSON)
        exhaustive_df.astype({
            "st_latency": float, 
            "mt_p_latency": float, 
            "mt_np_latency": float
        })

        cost_step_size = math.ceil(max_cost/4)
        if cost_step_size < 1:
            cost_step_size = 1
        print("Cost Step Size: {}, Max Cost: {}".format(cost_step_size, max_cost))
        fig, ax = plt.subplots(figsize=(14,7))
        for cost_index, cost_value in enumerate(range(cost_step_size, max_cost, cost_step_size)):
            print("Evaluating: ${}".format(cost_value))
            eval_count_vs_error_array = []
            for key in data:
                
                mhmrc_df = data[key][data[key]["cost"] == cost_value]
                mhmrc_df.astype({
                    "st_latency": float, 
                    "latency": float
                })
                ex_df = exhaustive_df[exhaustive_df["cost"] == cost_value]

                # key 0 has all the evaluations and key 1 has high error 
                if key <= 1:
                    continue 

                try:
                    assert(len(mhmrc_df)<=1 and len(ex_df) <= 1)
                except AssertionError:
                    print(mhmrc_df)
                    print(ex_df)
                    raise AssertionError

                if len(mhmrc_df) > 0 and len(ex_df) > 0:
                    mhmrc_row = mhmrc_df.iloc[0]
                    ex_row = ex_df.iloc[0]
                    ex_latency = ex_row[["st_latency", "mt_p_latency", "mt_np_latency"]].min(axis=0)
                    mhmrc_latency = mhmrc_row["latency"]
                    error = mhmrc_latency - ex_latency

                    if error < 0:
                        print(ex_row)
                        print(mhmrc_row)
                        raise ValueError("How is error negative?")

                    percent_error = 100*error/ex_latency
                    eval_count_vs_error_array.append([key, percent_error])
            
            sorted_plot_data = sorted(eval_count_vs_error_array ,key=lambda x: (x[0]))
            x_axis = np.array([_[0] for _ in sorted_plot_data], dtype=int)
            #scaled_x_axis = (x_axis - np.min(x_axis))/np.ptp(x_axis)
            ax.plot(x_axis, [_[1] for _ in sorted_plot_data], 
                "-{}".format(self.markerlist[cost_index]), markersize=15, alpha=0.75,
                markevery=math.ceil(len(sorted_plot_data)/10),
                label="${}".format(cost_value))
        
        ax.set_ylabel("Percent Error (%)", fontsize=30, labelpad=15)
        ax.set_xlabel("Evaluation Count", fontsize=30)
        ax.tick_params(axis='both', which='major', labelsize=25)
        plt.legend(fontsize=25)
        plt.tight_layout()
        output_path = self.per_workload_points_vs_error_plot_dir.joinpath("peval_vs_error-{}-{}-{}-{}.png".format(
            self.mt_label, write_policy, workload_name, self.step_size))
        print(output_path)
        print("Markevery: {}".format(math.ceil(len(sorted_plot_data)/10)))
        plt.savefig(str(output_path))
        plt.close()
