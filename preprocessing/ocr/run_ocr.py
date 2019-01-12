import os
from multiprocessing import Process
from process_queue import ProcessQueue

def main():
    process_queue = ProcessQueue(6)
    documentsDir = "./data/documents"
    outDir = "./data/ocr"
    os.makedirs(outDir, exist_ok=True)
    for rootdir, dirnames, fileNames in os.walk(documentsDir):
        if fileNames:
            outDirBaseText = os.path.join(outDir, "text", rootdir.split("\\")[-1])
            outDirBaseHOCR = os.path.join(outDir, "hocr", rootdir.split("\\")[-1])
            os.makedirs(outDirBaseText, exist_ok=True)
            os.makedirs(outDirBaseHOCR, exist_ok=True)
            process_queue.schedule_process(processBatch, (rootdir, fileNames, outDirBaseText, outDirBaseHOCR))
    process_queue.await_all_processes()
    print("----------------finished-----------------")

def processBatch(rootdir, fileNames, outDirBaseText, outDirBaseHOCR):
    print("processing directory " + rootdir)
    fileNumber = 1
    numFiles = len(fileNames)
    for fileName in fileNames:
        print("{} file: {}/{}".format(rootdir, fileNumber, numFiles))
        inputFilePath = os.path.join(rootdir, fileName)
        processFile(inputFilePath, os.path.join(outDirBaseText, fileName), os.path.join(outDirBaseHOCR, fileName))
        fileNumber += 1

def processFile(filePath, outDirText, outDirHOCR):
    if not os.path.isfile(outDirText + ".txt"):
        os.system(createTesseractCommand(filePath, outDirText, "text"))
    if not os.path.isfile(outDirHOCR + ".hocr"):
        os.system(createTesseractCommand(filePath, outDirHOCR, "hocr"))

def createTesseractCommand(inputfile, outputfile, mode):
    return "tesseract \"{}\" \"{}\" -l deu {}".format(inputfile, outputfile, mode)


if __name__ == "__main__":
    main()