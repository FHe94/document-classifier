import Levenshtein
import json
from ..utils import load_testdata_dictionary, save_dictionary, split_list, run_operation_parallel

OUT_DIR = r"C:\Users\felix\Documents\Repositories\anyreader\Classifier\data\dictionaries\from_testdata\distances.json"
NUM_PROCESSES = 6
MAX_RESULTS = 4

def measure_distances(split_words, all_words):
    distances = {}
    for i in range(len(split_words)):
        word = split_words[i]
        print("processing word {}/{}".format(i+1, len(split_words)))
        distances[word] = list(map(lambda e:e[0] ,get_lowest_n_distances(word, all_words, MAX_RESULTS)))

    return distances

def get_lowest_n_distances(source_word, all_words, max_results):
    distances = []
    for word in all_words:
        if word != source_word:
            distances.append([word, Levenshtein.distance(source_word, word)])
    distances.sort(key=lambda e: e[1])
    return distances[0:max_results]

def merge_dictionaries(dictionaries):
    result = {}
    for d in dictionaries:
        result = {**result, **d}
    return result

def main():
    words = load_testdata_dictionary()
    keys = list(words.keys())
    splits = split_list(keys, NUM_PROCESSES)
    print("measuring distances...")
    distances = run_operation_parallel(measure_distances, [ (split, keys) for split in splits ])
    print("merging results...")
    merged_results = merge_dictionaries(distances)
    print("saving results...")
    save_dictionary(merged_results, OUT_DIR)
    print("-----------done------------")

if __name__ == "__main__":
    main()