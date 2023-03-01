import whisper
import argparse
from timeit import default_timer
import torch
import json


def main():
    parser = argparse.ArgumentParser(
    description="Transcribe an audio file into text using whisper/openAi")
    parser.add_argument("filename",
                    help="Specify the audiofile that you want to transcribe")
    parser.add_argument("-m", "--modelsize", type=str, default="base",
                    help="Specify the model, can be: base, small, medium, large")
    parser.add_argument("-o", "--output", type=str, help="Define an output path for transcribed audio file")
    parser.add_argument("-json", type=bool, default=False, help="Output as a json file")
    parser.add_argument("-d", "--device", type=str, default="cpu", help="Specify use of GPU (option: cuda) or CPU (option cpu)")
    args = parser.parse_args()
    
    model = loadModel(args)
    transcribeAudio(args, model)


def loadModel(args):
    timeLoadModel = default_timer()
    if args.modelsize == "large" and args.device == "cuda":
        tmem = torch.cuda.get_device_properties(0).total_memory / 1024**2 
        if tmem < 10000:
            print(f"Not enough vram at GPU to load the large model. GPU vram: {tmem} MB. Need 10000 MB ´(10GB) to run. Setting device to CPU")
        
    model = whisper.load_model(args.modelsize)
    print(f"Model is loaded - modelsize = {args.modelsize} - Time to load: {round(default_timer() - timeLoadModel)} s. \n")
    return model

def transcribeAudio(args, model):
    print("Starting transcription")
    transcribeTime = default_timer()
    result = model.transcribe(args.filename, verbose= True)
    print(f"Files has been transcribet. - Time: {round(default_timer() - transcribeTime)} s. \n")


    if args.output:
        if args.json:
            outputjson = args.filename[:-4] + ".json"
            with open(outputjson, "w", encoding="utf-8") as fp:
                json.dump(result["segments"], fp, indent=4)
            print(f"Output saved to {outputjson}")   
        else:
            outputtxt = args.filename[:-4] + ".json"
            with open(outputtxt, "w", encoding="utf-8") as fp:
                fp.write(result["text"])
                print(f"Output saved to {outputtxt}")   
    else:
        print("Transcribed text \n")
        print(result["segments"])



if __name__ == '__main__':
    main()