pip install -r requirements.txt

wget "https://huggingface.co/lingbionlp/AIONER-0415/resolve/main/pretrained_models.zip" -O ./pretrained_models.zip
unzip ./pretrained_models.zip

# Remove the zip file

rm ./pretrained_models.zip