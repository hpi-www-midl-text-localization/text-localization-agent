# Instructions for training on COCO-Text

To train on the COCO-Text dataset, you need to download

* The basic COCO-training dataset from 2014. You can either download it from [this](http://cocodataset.org/#download)
website, or download it using [ChainerCV](https://chainercv.readthedocs.io/en/stable/reference/datasets.html#chainercv.datasets.COCOBboxDataset).
* The [COCO-Text](https://bgshih.github.io/cocotext/#h2-download) annotations.

After doing that, you have two options:

1. Rename the folder that contains the images to 'images' and place an image_locations.txt containing relative paths 
in this folder. For convenience, the txt is included in this repository.

2. Use the prepare_coco.py script to create a txt containing absolute image locations. This can then be placed 
anywhere independent of the images. Remember to then pass the `--absolute` parameter to the training script.

Now you can train on COCO-Text just like on any other dataset.
