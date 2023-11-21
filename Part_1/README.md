# Part I
![](../archive/result_img.png)
### Setup
```
conda create --name=part_1 python=3.9
conda activate part_1

# If you are using RTX 4090:
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# If you are using CPU only:
conda install pytorch torchvision

conda install -c conda-forge pythonocc-core=7.7.0
pip install -r requirements.txt
```
### Run
##### Edge Detector
```
# Tested on Windows platform.

• 3D -> 2D:
python main.py <input_folder>

• Remove duplicates:
# NOTE: backup your initial data before removing duplicates.
python main.py -rd 2 -rdm <hash or mse> <input_folder>

• Extra processing:
# The data that has undergone extra processing will be stored in the "out_ep" folder.
python main.py -epm <1 for padding-first, 2 for line-width-first> -pad <your_padding_size> -F <png or jpg> -W <desired_image_width> -H <desired_image_height> <input_folder>
```
##### Viewpoint Selector
```
Download ck.pth for ICNet: https://drive.google.com/drive/folders/1N3FSS91e7FkJWUKqT96y_zcsG9CRuIJw
Place "ck.pth" into the directory "models/saved_models/ck.pth".

# The selected data will be stored in the "out_vs" folder.
python main.py -vsm <1 for ICNet> <input_folder>
```
