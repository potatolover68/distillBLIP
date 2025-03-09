@echo off
echo Starting BLIP Distillation Pipeline
echo ==================================

REM Set Python path - update this if your Python installation is in a different location
set PYTHON_PATH=python

REM Activate virtual environment if you're using one
REM call path\to\your\venv\Scripts\activate.bat

REM Install requirements if needed
echo Installing requirements...
%PYTHON_PATH% -m pip install -r requirements.txt
%PYTHON_PATH% -m pip install pycocoevalcap

REM Run the download and training script
echo Running download and training script...
%PYTHON_PATH% scripts\download_and_train.py --subset --subset_size 5000 --num_epochs 5 --eval_after_train

echo Pipeline completed!
pause
