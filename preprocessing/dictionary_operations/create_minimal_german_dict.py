import Levenshtein
import os
import ..utils as utils


GERMAN_DICT_LOCATION = r"C:\Users\felix\Documents\Repositories\anyreader\Classifier\data\dictionaries\german\german.dic"
DICT_LOCATION = r"C:\Users\felix\Documents\Repositories\anyreader\Classifier\data\dictionaries\from_testdata\placeholders.json"
NUM_THREADS = 12

def main():
    german_dict = utils.load_german_dictionary(GERMAN_DICT_LOCATION)
    testdata_dict = utils.load_testdata_dictionary(DICT_LOCATION)

    splits = utils.split_list(list(testdata_dict.keys()), NUM_THREADS)

    minified_dicts = utils.run_operation_parallel(create_full_minified_dictionary, [ (split, german_dict) for split in splits ])
    merged_dict = create_merged_dict(minified_dicts)
    utils.save_dictionary(merged_dict, os.path.join(utils.DICTIONARIES_ROOT, "from_testdata", "german_full_min.json"))

def create_merged_dict(dicts):
    result_dict = dicts[0].copy()
    for i in range(1, len(dicts)):
        result_dict = { **result_dict, **dicts[i] }
    return list(result_dict.keys())

def create_minified_dictionary(words, german_dict):
    result_dict = {}
    num_words = len(words)
    for i in range(num_words):
        word = words[i]
        best_match = find_word_within_distance(word, german_dict, min(int(len(word)/2), 2))
        if best_match is not None:
            print("({} / {}): '{}' has match -----> '{}' ".format((i+1), num_words, word, best_match))
            result_dict[best_match] = 1
    return result_dict

def find_word_within_distance(source_word, german_dict, max_distance):
    normalized_source_word = source_word.strip().lower()
    if not is_valid_word(normalized_source_word):
        return None
    best_word = None
    best_distance = 500
    for key in german_dict:
        distance = Levenshtein.distance(normalized_source_word, key.strip().lower())
        if distance <= max_distance and distance < best_distance:
            best_distance = distance
            best_word = key
        if best_distance == 0:
            break
    return best_word

def create_full_minified_dictionary(words, german_dict):
    result_dict = {}
    num_words = len(words)
    for i in range(num_words):
        word = words[i]
        add_all_words_within_distance(word, result_dict, german_dict, min(int(len(word)/2), 2))
        if i % 100 == 0:
            print("Word {}/{}".format(i+1, num_words))
    return result_dict

def add_all_words_within_distance(source_word, target_dict, german_dict, max_distance):
    normalized_source_word = source_word.strip().lower()
    if is_valid_word(normalized_source_word):
        for key in german_dict:
            distance = Levenshtein.distance(normalized_source_word, key.strip().lower())
            if distance <= max_distance:
                target_dict[key] = 1

def is_valid_word(word):
    return len(word) > 2 and word.replace("-", "").isalpha()


if __name__ == "__main__":
    main()