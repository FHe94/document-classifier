import json
import multiprocessing.pool

DICTIONARIES_ROOT = r"C:\Users\felix\Documents\Repositories\anyreader\Classifier\data\dictionaries" 
GERMAN_DICT_PATH = r"C:\Users\felix\Documents\Repositories\anyreader\Classifier\data\dictionaries\german\german.dic"
TESTDATA_DICT_PATH = r"C:\Users\felix\Documents\Repositories\anyreader\Classifier\data\dictionaries\from_testdata\raw.json"

def load_testdata_dictionary(path = TESTDATA_DICT_PATH):
    content = ""
    with open(path, encoding="utf-8") as dictionaryFile:
        content = dictionaryFile.read()
    words = json.loads(content)
    return words

def load_german_dictionary(path = GERMAN_DICT_PATH):
    with open(path) as dictionary_file:
        result_dict = {}
        for line in dictionary_file.readlines():
            result_dict[line.strip().lower()] = 1
        return result_dict

def save_dictionary(dictionary, path):
    with open(path, "w", encoding="utf-8") as outfile:
        outfile.write(json.dumps(dictionary, ensure_ascii=False, indent=4))

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