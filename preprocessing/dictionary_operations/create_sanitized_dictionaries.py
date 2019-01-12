from ..utils import load_german_dictionary, load_testdata_dictionary, save_dictionary
import ..word_placeholders

PLACEHOLDERS = [ word_placeholders.DatePlaceholder(), word_placeholders.NumericPlaceholder() ]

OUT_PATH_NUMERICS = r"C:\Users\felix\Documents\Repositories\anyreader\Classifier\data\dictionaries\from_testdata\no_numerics.json"
OUT_PATH_PLACEHOLDERS = r"C:\Users\felix\Documents\Repositories\anyreader\Classifier\data\dictionaries\from_testdata\placeholders.json"

def main():
    #german_dict = load_german_dictionary()
    testdata_dict = load_testdata_dictionary()
    print("dictionary length: " + str(len(testdata_dict.keys())))
    print("creating dictionaries...")
    dict_stripped_special_chars = strip_special_characters(testdata_dict, "‚,:-|") #two different characters
    dict_no_numerics = remove_numeric_entries(dict_stripped_special_chars)
    placeholder_dict = insert_placeholders(dict_stripped_special_chars, PLACEHOLDERS)
    print("saving dictionaries")
    print("dictionary length: " + str(len(dict_no_numerics.keys())))
    save_dictionary(dict_no_numerics, OUT_PATH_NUMERICS)
    save_dictionary(placeholder_dict, OUT_PATH_PLACEHOLDERS)
    print("done")


def strip_special_characters(dictionary, chars):
    print("removing special characters...")
    "".strip
    result_dict = {}
    for key in dictionary.keys():
        result_dict[key.strip(chars)] = dictionary[key]
    return result_dict

def remove_numeric_entries(dictionary):
    print("removing numeric entries...")
    result_dict = {}
    for key in dictionary.keys():
        if not isNumeric(key):
            result_dict[key] = dictionary[key]
    return result_dict

def isNumeric(string):
    split_chars = ".-/+:'\\|,*°"
    currencies = "€$£"
    cleaned_string = string
    for char in split_chars + currencies:
        cleaned_string = cleaned_string.replace(char, "")
    return cleaned_string.isnumeric()

def insert_placeholders(dictionary, placeholders):
    print("inserting placeholders...")
    result_dict = {}
    for key, value in dictionary.items():
        label = test_placeholders(key, placeholders)
        if label is not None:
            print("replacing {} by {}".format(key, label))
            insert_key(result_dict, label) 
        else:
            result_dict[key] = value
    return result_dict

def test_placeholders(key, placeholders):
    label = None
    i = 0
    while i < len(placeholders) and label is None:
        placeholder = placeholders[i]
        if placeholder.test_string(key):
            label = placeholder.Label
        i += 1
    return label

def insert_key(dictionary, key):
    if key in dictionary:
        dictionary[key] += 1
    else:
        dictionary[key] = 1

if __name__ == "__main__":
    main()