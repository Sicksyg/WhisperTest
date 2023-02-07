"# WhisperTest" 
## Setup Whisper:

### 1.)
download python 3.8-3.10 whithout setting it to PATH, Notice the path of the install. 

### 2.)
Make a virtual enviroment with the param:

virtualenv -p <path/to/Python3X/python.exe> <path/to/venv>

Example:
virtualenv -p C:\Users\SICK\AppData\Local\Programs\Python\Python310\python.exe C:\Users\SICK\Desktop\Transcript\venv

Activate the enviroment
python nameOfVenv\Scripts\Activate.bat


### 3.)
Download the CUDAtoolkit (11.6 or 11.7) install it without video drivers and nvidia geforce


### 4.)
Clone the Whisper repo from github
- Install git if not installed
pip install git+https://github.com/openai/whisper.git 


### 5.)
Update pytorch with the corresponding CUDAToolKit (DO THIS AFTER CLONING WHISPER!)

pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117


### 6.) Use my baller script B-)

