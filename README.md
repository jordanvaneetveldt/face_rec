# face_rec
An eye-blink detection-based face liveness detection algorithm to thwart photo attacks.

# Rquirements

## dlib

After **others**, install dlib first. If pip install gives build problems, then it is best to install cmake using Anaconda.

```sh
   conda activate <your_env_name>
   conda install cmake
```

## OpenCV

It is a must requirement for its ability to work with faces.

## face_recognition

Just pip install this library. No available on conda repo yet.

```sh
   pip install face_recognition
```
## keras

Install keras. If you want GPU acceleration enabled then install keras-gpu in Anaconda. Also to enable cuda support you need to install cuda from nvidia's website. Also install cuDNN. This will also enable cuda in dlib while building.

## tqdm

Just pip install this library for progress bar.

## Python Imaging Library

```sh
   pip install PIL
```
## Others - not optional

- scipy
- numpy