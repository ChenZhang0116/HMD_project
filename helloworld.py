import pandas as pd
from pathlib import Path
from PIL import Image
import torchvision
import torch
import matplotlib.pyplot as plt
from pandas_path import path

data_dir = Path.cwd().parent / "data"
img_path = data_dir / "img"
train_path = data_dir / "train.jsonl"
dev_unseen_path = data_dir / "dev_unseen.jsonl"
test_unseen_path = data_dir / "test_unseen.jsonl"
dev_seen_path = data_dir / "dev_seen.jsonl"
test_seen_path = data_dir / "test_seen.jsonl"

# load descriptor list of training samples + print statistics:
train_samples_frame = pd.read_json(train_path, lines=True)
train_samples_frame = train_samples_frame.reset_index(drop=True)
train_samples_frame.img = train_samples_frame.apply(lambda row: (data_dir/row.img), axis=1)

allImgExist = train_samples_frame.img.path.exists().all()

#
#
#
# print(train_samples_frame.head())
# print(train_samples_frame.label.value_counts())
# print(train_samples_frame.text.map(
#     lambda text: len(text.split(" "))).describe())
#
# # visualize sizes of several images:
# images = [Image.open(
#     data_dir / train_samples_frame.loc[i, "img"]
# ).convert("RGB")
#           for i in range(5)
#           ]
# for im in images:
#     print(im.size)
#
# # define a callable image_transform with Compose
# image_transform = torchvision.transforms.Compose(
#     [
#         torchvision.transforms.Resize(size=(224, 224)),  # this function interpolates when needed so may distort images
#         torchvision.transforms.ToTensor()
#     ])
# # convert the images and prepare for visualization
# tensor_img = torch.stack(  # make a single tensor object out of them
#     [image_transform(image) for image in images]
# )
# grid = torchvision.utils.make_grid(tensor_img)
# # plot
# plt.rcParams["figure.figsize"] = (20, 5)
# plt.axis('off')
# plt.imshow(grid.permute(1, 2, 0))
# plt.show()
