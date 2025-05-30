hdir=$(pwd)
cd ..

mkdir datasets

cd datasets

kaggle competitions download -c pii-detection-removal-from-educational-data
unzip pii-detection-removal-from-educational-data.zip -d pii-detection-removal-from-educational-data
rm pii-detection-removal-from-educational-data.zip

kaggle datasets download -d conjuring92/pii-mix-v01
unzip pii-mix-v01.zip -d ./pii_mix_v01
rm pii-mix-v01.zip

cd $hdir