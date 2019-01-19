class Dictionary:

    def __init__(self, values):
        self.__values = values.copy()
        self.__unknown_word_index = values["<unknown>"]
        self.__length = len(values.keys())

    def get_token_list(self):
        return list(self.__values.keys())

    def find_similar_word(self, word, max_distance):
        best_word = None
        smallest_distance = 5000
        for value in self.__values:
            distance = Levenshtein.distance(word, value)
            if distance == 0:
                return value
            elif distance < smallest_distance:
                smallest_distance = distance
                best_word = value
        print("replacing {} with {}".format(word, best_word))
        return best_word

    def get_word_index(self, word):
        return self.__values.get(word, self.__unknown_word_index)

    def get_values(self):
        return self.__values

    def get_length(self):
        return self.__length