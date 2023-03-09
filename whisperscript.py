import whisper
import argparse
from timeit import default_timer
import torch
import json
import os


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe an audio file into text using whisper/openAi")
    parser.add_argument("-f", "--filename", type=str,
                        help="Specify the audiofile that you want to transcribe")
    parser.add_argument("-m", "--modelsize", type=str, default="large-v2",
                        help="Specify the model, can be: base, small, medium, large-v2")
    parser.add_argument("-o", "--output", type=str,
                        help="Define an output path for transcribed audio file")
    parser.add_argument("-json", type=bool, default=False,
                        help="Output as a json file")
    parser.add_argument("-d", "--device", type=str, default="cuda",
                        help="Specify use of GPU (option: cuda) or CPU (option: cpu)")
    parser.add_argument("-b", "--batch", type=bool, default=False, help="If True, the transcriber takes all the files in inport and transcribe them")
    args = parser.parse_args()

    model = loadModel(args)
    transcribeAudio(args, model)


def loadModel(args):
    timeLoadModel = default_timer()
    if args.modelsize == "large-v2" and args.device == "cuda":
        tmem = torch.cuda.get_device_properties(0).total_memory / 1024**2
        if tmem < 10000:
            print(
                f"Not enough vram at GPU to load the large model. GPU vram: {tmem} MB. Need 10000 MB ´(10GB) to run. Setting device to CPU")

    model = whisper.load_model(args.modelsize)
    print(
        f"Model is loaded - modelsize = {args.modelsize} - Time to load: {round(default_timer() - timeLoadModel)} s. \n")
    return model


def transcribeAudio(args, model):
    import_path = "./Import/" 
    export_path = "./Export/"

    print("Starting transcription")
    if args.batch == True:
        for f in os.listdir(import_path):
            transcribeTime = default_timer()
            result = model.transcribe(f, verbose=False)
            filename = f
            file_printer(args, result, filename)
    else:
        transcribeTime = default_timer()
        result = model.transcribe(args.filename, verbose=False)
        filename = args.filename.rsplit("/")[-1]
        file_printer(args, result, filename)
    print(f"Files has been transcribet. - Time: {round(default_timer() - transcribeTime)} s. \n")
    
"""Printer for results"""
def file_printer(args, result, filename):
    if args.output is not None:  # If the output options has been added to the argument
        outputtxt = args.output + "/" + filename[:-4] + ".txt"
        outputjson = args.output + "/" + filename[:-4] + ".json"

        if args.json is True:  # If -json == True
            # Writes json file
            with open(outputjson, "x", encoding="utf-8") as fp:
                json.dump(result["segments"], fp, indent=4)
        # Writes txt file
            with open(outputtxt, "w", encoding="utf-8") as fp:
                fp.write(result["text"])

            print(f"Output saved to {outputjson}")
            print(f"Output saved to {outputtxt}")

        else:  # Only save a txt file
            # Writes txt file
            with open(outputtxt, "w", encoding="utf-8") as fp:
                fp.write(result["text"])
                print(f"Output saved to {outputtxt}")
    else:  # Don't save file, only display results in terminal
        print("Transcribed text \n")
        print(result["text"])


if __name__ == '__main__':
    main()
