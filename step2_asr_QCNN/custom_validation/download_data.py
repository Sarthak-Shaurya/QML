import os
import urllib.request
import tarfile

data_dir = 'data/binary_speech'
os.makedirs(data_dir, exist_ok=True)

url = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
file_path = "speech_commands.tar.gz" 

print("Downloading Google Speech Commands V2 (2.3 GB). This will take a few minutes...")
urllib.request.urlretrieve(url, file_path)
print("Download complete. Extracting 'left' and 'right' folders...")

# A much more robust matching condition
with tarfile.open(file_path, "r:gz") as tar:
    for member in tar.getmembers():
        # Catch the folder whether it's "left/...", "./left/...", or "dataset/left/..."
        path_parts = member.name.split('/')
        if 'left' in path_parts or 'right' in path_parts:
            tar.extract(member, path=data_dir)

print("Smart extraction complete. Deleting the massive 2.3 GB zip file...")
os.remove(file_path)
print("Success! Your binary dataset is securely in 'data/binary_speech'.")