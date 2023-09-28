#!/bin/bash

env_name=imitator
conda remove -n $env_name --all

echo "Creating conda environment"
conda create -n $env_name python=3.8.5
eval "$(conda shell.bash hook)" # make sure conda works in the shell script
conda activate $env_name
if echo $CONDA_PREFIX | grep $env_name
then
    echo "Conda environment successfully activated"
else
    echo "Conda environment not activated. Probably it was not created successfully for some reason. Please activate the conda environment before running this script"
    exit
fi

echo "Installing conda packages"
echo "Installing pytorch"
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=10.2 -c pytorch

conda env update -n $env_name --file environment.yml
pip install smplx
echo "Installation finished"
