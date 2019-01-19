import os
import argparse
from utils.utils import run_operation_parallel
from preprocessing.ocr.ocr_engine import TesseractOcrEngine

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    process_args = [ ]
    for rootdir, dirnames, filenames in os.walk(args.documents_dir):
        if filenames:
            out_dir_base = os.path.join(args.out_dir, rootdir.split("\\")[-1])
            os.makedirs(out_dir_base, exist_ok=True)
            process_args.append((rootdir, filenames, out_dir_base, args.format))
    run_operation_parallel(process_batch, process_args)
    print("----------------finished-----------------")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("documents_dir", metavar="d")
    parser.add_argument("out_dir", metavar="o")
    parser.add_argument("--format", "-f", choices = [ "text", "hocr"], default="text")
    return parser.parse_args()

def process_batch(rootdir, filenames, out_dir_base, output_format):
    print("processing directory " + rootdir)
    ocr_engine = TesseractOcrEngine()
    file_counter = 1
    num_files = len(filenames)
    for filename in filenames:
        print("{} file: {}/{}".format(rootdir, file_counter, num_files))
        input_file_path = os.path.join(rootdir, filename)
        process_file(ocr_engine, input_file_path, os.path.join(out_dir_base, filename), output_format)
        file_counter += 1

def process_file(ocr_engine, filepath, out_dir_base, output_format):
    output_extension = ".txt" if output_format == "text" else ".hocr"
    if not os.path.isfile(out_dir_base + output_extension):
        ocr_engine.run_ocr(filepath, out_dir_base, output_format)

if __name__ == "__main__":
    main()