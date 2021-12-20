from mascots.blockReader.CPReader import CPReader

r = CPReader("/research2/mtc/cp_traces/csv_traces/w105.csv")

print(r.get_next_block_req())