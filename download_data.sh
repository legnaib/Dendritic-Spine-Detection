echo "Starting downloading data..."
wget "https://drive.google.com/uc?export=download&id=1r9bGOeOxEd6Clg8Sw5ajcxLO-C_xC9K8" -O data.zip # -o data.zip
echo "Finished download. Unzipping..."
unzip data.zip -d .tmp
echo "Moving data to folders"
mkdir -p data/raw/
mv -f .tmp/data/raw/* data/raw/
mkdir -p own_models/default_model/
mv -f .tmp/own_models/default_model/* own_models/default_model/
rm -rf .tmp
rm data.zip
echo "Finished"