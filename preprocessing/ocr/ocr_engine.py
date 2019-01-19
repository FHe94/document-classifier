import abc
import os

class TesseractOcrEngine:
    def run_ocr(self, input_file_path, output_file_path, output_format = "text", language = "deu"):
        return os.system(self.__createTesseractCommand(input_file_path, output_file_path, output_format, language))

    def __createTesseractCommand(self, input_file_path, output_file_path, output_format, language):
        return "tesseract \"{}\" \"{}\" -l {} {}".format(input_file_path, output_file_path, language, output_format)