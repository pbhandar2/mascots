import argparse, itertools, time, math
import multiprocessing as mp

from CostAnalysis import CostAnalysis

def run_cost_analysis(workload_name, cost):
    print("workload {}, cost {}".format(workload_name, cost))
    analysis = CostAnalysis(workload_name, cost)
    analysis.run()


if __name__=='__main__':
    parser = argparse.ArgumentParser(
                description="Spawn processing for each cost value for a given workload")
    parser.add_argument("workload_name", help="The name of the workload to be evaluted")
    parser.add_argument("start_cost", type=int, help="The start cost limit of the cache ")
    parser.add_argument("end_cost", type=int, help="The start cost limit of the cache ")
    args = parser.parse_args()
    
    start = time.perf_counter()
    cost_array = list(range(args.start_cost, args.end_cost+1))
    arg_list = list(itertools.product([args.workload_name], cost_array))

    with mp.Pool(processes=args.end_cost-args.start_cost) as pool:
        results = pool.starmap(run_cost_analysis, arg_list)
    end = time.perf_counter()
    print("Workload {}, Time {}".format(args.workload_name, end-start))