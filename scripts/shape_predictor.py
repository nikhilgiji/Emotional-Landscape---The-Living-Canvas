import urllib.request
import shutil
import tarfile

url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
file_name = "shape_predictor_68_face_landmarks.dat.bz2"

# Download the file
urllib.request.urlretrieve(url, file_name)

# Extract the file
with tarfile.open(file_name, 'r:bz2') as archive:
    archive.extractall()

# Remove the compressed file
shutil.rmtree(file_name)
