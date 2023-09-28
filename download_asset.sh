echo "In order to run Imitator, you need to download FLAME. Before you continue, you must register and agree to license terms at:"
echo -e '\e]8;;https://flame.is.tue.mpg.de\ahttps://flame.is.tue.mpg.de\e]8;;\a'

while true; do
    read -p "I have registered and agreed to the license terms at https://flame.is.tue.mpg.de? (y/n)" yn
    case $yn in
        [Yy]* ) break;;
        [Nn]* ) exit;;
        * ) echo "Please answer yes or no.";;
    esac
done

echo "Downloading assets to run Imitator..."

echo "Downloading Imitator pretrained model..."
wget "https://keeper.mpdl.mpg.de/f/24e17fa78a4e4830b44a/?dl=1" -O pretrained_model.zip
echo "Extracting pretrained_model..."
unzip pretrained_model.zip
rm pretrained_model.zip

flame_path=FLAMEModel
echo "Downloading FLAME related assets"
rm -rf $flame_path
mkdir -p $flame_path
wget "https://keeper.mpdl.mpg.de/f/cf406e4f145f484a958a/?dl=1" -O FLAME.zip
echo "Extracting FLAME..."
unzip FLAME.zip -d $flame_path
rm FLAME.zip

echo "Assets for EMOCA downloaded and extracted."
