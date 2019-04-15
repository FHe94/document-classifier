import json
import multiprocessing.pool
import os

def split_number_into_integers(number, num_integers):
    division_float = float(number) / num_integers
    integers = []
    rest = 0
    for i in range(num_integers):
        integer = round(division_float + rest)
        rest = division_float + rest - integer
        integers.append(integer)
    return integers

def split_list(target_list, num_splits):
    splits = []
    startindex = 0
    num_elements = split_number_into_integers(len(target_list), num_splits)
    for num_elements_per_split in num_elements:
        endindex = startindex + num_elements_per_split
        splits.append(target_list[startindex:endindex])
        startindex=endindex
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

def save_json_file(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(content, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=4)

def read_json_file(path):
    with open(path, encoding = "utf-8") as json_file:
        return json.load(json_file, encoding="utf-8")

def try_parse_json_config(parse_function, config_path):
    try:
        return parse_function(config_path)
    except KeyError as e:
        raise Exception('Unable to load config. Key "{}" not found in file "{}"'.format(
            e.args[0], config_path))

def get_resource_by_condition(scope, condition):
    for name, attribute in scope.items():
        if condition(name, attribute):
            return attribute