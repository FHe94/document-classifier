import json
import multiprocessing.pool

def split_list(target_list, num_splits):
    num_entries = len(target_list)
    entries_per_split = int(num_entries/num_splits)
    splits = []
    for i in range(num_splits):
        startindex = i*entries_per_split
        endindex = (i+1)*entries_per_split if i < num_splits - 1 else num_entries
        splits.append(target_list[startindex:endindex])
    return splits

def run_operation_parallel(operation, arg_sets, num_processes=12):
    results = []
    with multiprocessing.pool.Pool(num_processes) as process_pool:
        try:
            for args in arg_sets:
                results.append(process_pool.apply_async(operation, args))
            process_pool.close()
        except:
            process_pool.terminate()
        process_pool.join()
    return [ result.get(1) for result in results ]