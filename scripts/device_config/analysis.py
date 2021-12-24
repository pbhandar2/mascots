import sys 
import pathlib 
from mascots.experiments.DeviceAnalysis import DeviceAnalysis 
 
if __name__ == "__main__":
    d_analysis = DeviceAnalysis()
    d_analysis.load_system_data("/home/pranav/mtc/system_measure/local_hdd")