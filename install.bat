@echo off

set env_name=imitator
CALL conda remove -n %env_name% --all

echo Creating conda environment
CALL conda create -n %env_name% python=3.8.5
CALL conda activate %env_name%

echo %CONDA_PREFIX% | findstr /C:%env_name% >nul
if errorlevel 1 (
    echo Conda environment not activated. Probably it was not created successfully for some reason. Please activate the conda environment before running this script
    exit 1
) else (
    echo Conda environment successfully activated
)


echo Installing conda packages
echo Installing pytorch
call conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=10.2 -c pytorch

call conda env update -n %env_name% --file environment.yml
call pip install smplx
echo Installation finished
