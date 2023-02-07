import whisper
import torch

def transcribeAudio(modelSize, audioFile):
    if audioFile == "audio1":
        audio = "./Interview_20_Danske Bank.m4a"
    if audioFile == "audio2":
        audio = "./whispertest.mp3"
    # modelSize = "small"

    if torch.cuda.is_available() == True:
        print(f"Cuda device {torch.cuda.get_device_name()} is avaliable. Setting DEVICE to {torch.cuda.get_device_name()}")
        
        torch.cuda.empty_cache()
        #torch.cuda.set_per_process_memory_fraction(0.1)
        DEVICE = 'cuda'

    print()
    model = whisper.load_model(modelSize, device = DEVICE, in_memory=False)
    result = model.transcribe(audio)
    print(result["text"])

transcribeAudio("large", "audio2")