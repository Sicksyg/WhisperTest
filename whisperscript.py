import whisper
import argparse
from timeit import default_timer
import torch

def main():
    parser = argparse.ArgumentParser(
    description="Transcribe an audio file into text using whisper/openAi")
    parser.add_argument("filename",
                    help="Specify the audiofile that you want to transcribe")
    parser.add_argument("-m", "--modelsize", type=str, default="base",
                    help="Specify the model, can be: base, small, medium, large")
    parser.add_argument("-o", "--output", type=str, help="Define an output path for transcribed audio file")
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
    print(f"Model is loaded, modelsize= {args.modelsize} Time: {round(default_timer() - timeLoadModel)} s. \n")
    return model

def transcribeAudio(args, model):
    transcribeTime = default_timer()
    result = model.transcribe(args.filename)
    print(f"Files has been transcribet. Time: {round(default_timer() - transcribeTime)} s. \n")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fp:
            fp.write(result["text"])
            print(f"Output saved to {args.output}")   
    else:
        print("Transcribed text \n")
        print(result["text"])

if __name__ == '__main__':
    main()