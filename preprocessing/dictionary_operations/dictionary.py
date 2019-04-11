class Dictionary:

    __unknown_word_placeholder = "<unknown>"

    def __init__(self, values):
        self.__values = values.copy()

        if self.__unknown_word_placeholder in values:
            self.__unknown_word_index = values[self.__unknown_word_placeholder]
        else:
            self.__values[self.__unknown_word_placeholder] = len(values)
        
        self.__length = len(self.__values)

    def __contains__(self, key):
        return key in self.__values

    def __len__(self):
        return self.__length

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
        return best_word

    def get_word_index(self, word):
        return self.__values.get(word, self.__unknown_word_index)

    def get_values(self):
        return self.__values