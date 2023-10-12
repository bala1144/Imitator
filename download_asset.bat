@echo off

echo In order to run Imitator, you need to download FLAME. Before you continue, you must register and agree to license terms at: https://flame.is.tue.mpg.de

:A
choice /C yn  /M "I have registered and agreed to the license terms at https://flame.is.tue.mpg.de"
if %errorlevel%==2 goto A

echo Unzip the mmbp weights
unzip assets/release/mbp_weights_win03_weight05.zip -d assets/release

echo Downloading assets to run Imitator...

echo Downloading Imitator pretrained model...
wget "https://keeper.mpdl.mpg.de/f/24e17fa78a4e4830b44a/?dl=1" -O pretrained_model.zip
echo Extracting pretrained_model...
unzip pretrained_model.zip
del pretrained_model.zip

set flame_path=FLAMEModel
echo Downloading FLAME related assets
mkdir %flame_path%
wget "https://keeper.mpdl.mpg.de/f/cf406e4f145f484a958a/?dl=1" -O FLAME.zip
echo Extracting FLAME...
unzip FLAME.zip -d %flame_path%
del FLAME.zip

echo Assets for EMOCA downloaded and extracted.
