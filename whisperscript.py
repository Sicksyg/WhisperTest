import whisper
import argparse
from timeit import default_timer
import torch
import json
import os
import logging

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe an audio file into text using whisper/openAi")
    parser.add_argument("-f", "--filename", type=str,
                        help="Specify the audiofile that you want to transcribe")
    parser.add_argument("-m", "--modelsize", type=str, default="large-v2",
                        help="Specify the model, can be: base, small, medium, large-v2")
    parser.add_argument("-o", "--output", type=str,
                        help="Define an output path for transcribed audio file")
    parser.add_argument("-json", action="store_true",
                        help="Output as a json file")
    parser.add_argument("-ts", action="store_true",
                        help="Output as a timestamped")
    parser.add_argument("-d", "--device", type=str, default="cuda",
                        help="Specify use of GPU (option: cuda) or CPU (option: cpu)")
    parser.add_argument("-b", "--batch", action="store_true", help="If True, the transcriber takes all the files in inport and transcribe them")
    args = parser.parse_args()

    model = loadModel(args)
    if args.batch:
        batch_process(args, model)
    else:
        transcribeAudio(args, model)

def timer(func):
    def timethings(*args, **kwargs):
        start = default_timer()
        func(*args, **kwargs)
        end = default_timer()
        print(f"Operation '{func.__name__}' took {round(end - start)} seconds.")
    return timethings


def loadModel(args):
    msize = args.modelsize
    timeLoadModel = default_timer()
    if args.modelsize == "large-v2" and args.device == "cuda":
        tmem = torch.cuda.get_device_properties(0).total_memory / 1024**2
        if tmem < 10000:
            logging.warning(f"Not enough vram at GPU to load the large model. GPU vram: {tmem} MB. Need 10000 MB ´(10GB) to run.")
            msize = "medium"

    model = whisper.load_model(msize)
    logging.info(f"Model is loaded - modelsize = {args.modelsize} - Time to load: {round(default_timer() - timeLoadModel)} s.")
    logging.info(f"Device: {model.device}")
    return model


def transcribeAudio(args, model):
    logging.info("Starting transcription \n")
    transcribeTime = default_timer()
    result = model.transcribe(args.filename, verbose=False)
    filename = args.filename.rsplit("/")[-1]
    file_printer(args, result, filename)
    logging.info(f"Files has been transcribet. - Time: {round(default_timer() - transcribeTime)} s. \n")


def batch_process(args, model):
    logging.info("Starting batch transcription \n")
    import_path = "./Import/" 
    export_path = "./Export/"
    for file in os.listdir(import_path):
        print(f"files in import folder {os.listdir(import_path)} \n")
        result = model.transcribe(file, verbose=False)
        file_printer(args, result, file)

"""Printer for results"""

def file_printer(args, result, filename):
    if args.output is not None:  # If the output options has been added to the argument
        output_txt = args.output + "/" + filename[:-4] + ".txt"
        output_json = args.output + "/" + filename[:-4] + ".json"
        output_ts = args.output + "/" + filename[:-4] + ".txt"

        if args.json:  # If -json == True
            # Writes json file
            with open(output_json, "x", encoding="utf-8") as fp:
                json.dump(result["segments"], fp, indent=4)

            # Writes txt file
            with open(output_txt, "w", encoding="utf-8") as fp:
                fp.write(result["text"])

            logging.info(f"Output saved to {output_json}")
            logging.info(f"Output saved to {output_txt}")
        
        if args.ts:
            result["segments"]
            with open(output_json, "x", encoding="utf-8") as fp:
                json.dump(result["segments"], fp, indent=4)

        else:  # Only save a txt file
            # Writes txt file
            with open(output_txt, "w", encoding="utf-8") as fp:
                fp.write(result["text"])
                logging.info(f"Output saved to {output_txt}")
    else:  # Don't save file, only display results in terminal
        print("\n Transcribed text: \n")
        print(result["text"] + "\n")


if __name__ == '__main__':
    main()
