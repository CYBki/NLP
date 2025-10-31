import os
import shutil
import kagglehub

# Download latest version
path = kagglehub.dataset_download("uciml/sms-spam-collection-dataset")

# Your target directory
target_dir = "/home/syorgun/NLP/datasets/sms_spam"

# create folder if not exists
os.makedirs(target_dir, exist_ok=True)

# copy all downloaded files to your own directory
for file in os.listdir(path):
    shutil.copy(os.path.join(path, file), target_dir)

print("Files copied to:", target_dir)
