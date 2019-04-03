import os
from tempfile import mkstemp
from nltk.tokenize import word_tokenize
from .processing_steps.processing_steps import FilterTokens, WordPlaceholders, Normalize, Stemming
from .ocr.ocr_engine import TesseractOcrEngine

class DocumentProcessor:

    def __init__(self, name, processing_steps):
        self.name = name
        self.__processing_steps = processing_steps
        self.__ocr_engine = TesseractOcrEngine()

    def process_image_document(self, document_path):
        with TempTextFile() as outpath:
            self.__ocr_engine.run_ocr(document_path, os.path.splitext(outpath)[0], "text")
            return self.process_text_document(outpath)

    def process_text_document(self, document_path):
        document_content = self.__load_document(document_path)
        return self.__get_tokens(document_content)

    def process_tokens(self, tokens):
        return self.__apply_processing_steps(tokens)

    def __get_tokens(self, document_content):
        tokens = word_tokenize(document_content, language="german")
        return self.process_tokens(tokens)

    def __apply_processing_steps(self, tokens):
        processed_tokens = tokens
        if self.__processing_steps:
            for processing_step in self.__processing_steps:
                processed_tokens = processing_step.process(processed_tokens)
        return processed_tokens

    def __load_document(self, document_path):
        with open(document_path, encoding="utf-8") as document_file:
            return document_file.read()

class TempTextFile:

    def __enter__(self):
        filehandle, self.__path = mkstemp(suffix=".txt")
        os.close(filehandle)
        return self.__path

    def __exit__(self, arg1, arg2, arg3):
        os.remove(self.__path)