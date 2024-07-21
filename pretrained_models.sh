$base_url = "https://huggingface.co/lingbionlp/AIONER-0415/resolve/main/pretrained_models.zip"

# Download the file & unzip

wget $base_url
unzip pretrained_models.zip

# Remove the zip file

rm pretrained_models.zip

# List the files in the directory

ls pretrained_models