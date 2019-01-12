import abc
import re

class PlaceholderWord:

    def __init__(self):
        self.Label = ""

    @abc.abstractmethod
    def test_string(string):
        return

class DatePlaceholder(PlaceholderWord):

    __separator = r"[\./-]"
    __day = r"\d{1,2}"
    __month = r"(\d{1,2}|januar|februar|märz|april|mai|juni|juli|august|september|oktober|november|dezember)"
    __year = r"(\d{2}|\d{4})"
    __pattern = r"^{0}{3}{1}{3}{2}$".format(__day, __month, __year, __separator)
    
    def __init__(self):
        self.Label = "<date>"

    def test_string(self, string):
        return re.match(self.__pattern, string.strip(), flags=re.I) is not None

class TelphonePlaceholder(PlaceholderWord):

    __prefix = r"(0|\+49|0049)"
    __area_code = r"\d{4,5}"
    __tel_number = r"\d{3}\d*"
    __pattern = r"^{0}\s*{1}[\s/]*{2}$".format(__prefix, __area_code, __tel_number)

    def __init__(self):
        self.Label = "<tel>"

    def test_string(self, string):
        return re.match(self.__pattern, string.strip(), flags=re.I) is not None

class NumericPlaceholder(PlaceholderWord):

    __currencies = "€$£"
    __special_chars = ".-/+:'\\|,*°"
    __chars_to_remove = __currencies + __special_chars

    def __init__(self):
        self.Label = "<numeric>"

    def test_string(self, string):
        cleaned_string = string
        for char in self.__chars_to_remove:
            cleaned_string = cleaned_string.replace(char, "")
        return cleaned_string.isnumeric()