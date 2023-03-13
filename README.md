# WhisperTest

This is a "fork" of the openai/whisper repo ```https://github.com/openai/whisper```. This repo is an implementation for use in academic translation, with some small tweaks and personal CLI agent. 

## Setup Whisper:

### 1.)
download python 3.8-3.10 whithout setting it to PATH, Notice the path of the install. 

### 2.)
Make a virtual enviroment with the param:

```virtualenv -p <path/to/Python3X/python.exe> <path/to/venv>```

Example:
```virtualenv -p C:\Users\SICK\AppData\Local\Programs\Python\Python310\python.exe C:\Users\SICK\Desktop\Transcript\venv```

Activate the enviroment:
```python nameOfVenv\Scripts\Activate.bat```


### 3.)
Download:
1. The CUDAtoolkit (11.6 or 11.7) install it without video drivers and nvidia geforce
2. Ffmpeg CLI from essensials (https://git.ffmpeg.org/ffmpeg.git or Win binaries from https://www.gyan.dev/ffmpeg/builds/)


### 4.)
Clone the Whisper repo from github
- Install git if not installed
```pip install git+https://github.com/openai/whisper.git```


### 5.)
Update pytorch with the corresponding CUDAToolKit (DO THIS AFTER CLONING WHISPER!)

First uninstall PyTorch
```
pip3 uninstall torch torchvision torchaudio
```

Then purge the pip cache
```
pip3 cache purge
```

Then install PyTorch again
From: https://pytorch.org/get-started/locally/
```
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
```

### 6.)
OPTIONAL: Install jupyter and a ipython kernel
```
pip install jupyter
pip install ipython
ipython kernel install --user --name=<name of venv>
```

### 7.)
Use my baller script B-)
```
>>> python3.10 Whisper_cli.py -h

usage: Whisper_cli.py [-h] [-f FILENAME] [-m MODELSIZE] [-o OUTPUT] [-json] [-ts] [-d DEVICE] [-b]

Transcribe an audio file into text using whisper/openAi

options:
  -h, --help            show this help message and exit
  -f FILENAME, --filename FILENAME
                        Specify the audiofile that you want to transcribe
  -m MODELSIZE, --modelsize MODELSIZE
                        Specify the model, can be: base, small, medium, large-v2
  -o OUTPUT, --output OUTPUT
                        Define an output path for transcribed audio file
  -json                 Output as a json file
  -ts                   Output as a timestamped
  -d DEVICE, --device DEVICE
                        Specify use of GPU (option: cuda) or CPU (option: cpu)
  -b, --batch           If set, the transcriber takes all the files in inport and transcribe them
```

