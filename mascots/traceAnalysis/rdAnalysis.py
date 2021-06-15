import pandas as pd 
from plot import plot_rd_hist

class RDAnalysis:

    def __init__(self, rd_path):
        self.path = rd_path 
        self.data = self.load_data()


    def load_data(self):
        return pd.read_csv(self.path, names=["rd", "op"])


    def get_mhrc_exclusive(self):

        # generate a multi-tier hit rate function from an rd file 
        

        # take the rd file and generate a hit rate curve 


        # for exclusive cache the hit rate curve for all the value that sum
        # to the same cache size is the same 

        
        pass


    def get_mhrc_inclusive(self):

        # generate a multi-tier hit rate function from an rd file 

        pass 


    def plot_rd_hist(self):

        pass 
