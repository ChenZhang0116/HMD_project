import pandas as pd
from pathlib import Path
import torch
import cv2

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
train_samples_frame.img = train_samples_frame.apply(lambda row: (data_dir / row.img), axis=1)
N_image = 5
image_list = [cv2.cvtColor(cv2.imread(str(data_dir / train_samples_frame.loc[i, "img"])), cv2.COLOR_RGB2BGR)
              for i in range(N_image)]
text_list = [train_samples_frame.loc[i, "text"] for i in range(N_image)]

model_dir = Path.cwd().parent / "downloaded_models"


from transformers import BertTokenizer, VisualBertModel
from get_visual_embeddings import get_visual_embeddings

vb_model = VisualBertModel.from_pretrained(model_dir / "uclanlp-visualbert-nlvr2-coco-pre")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

inputs = tokenizer(text_list, padding=True, truncation=True, return_tensors="pt")
# this is a custom function that returns the visual embeddings given the image path
cfg_path = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
visual_embeds = get_visual_embeddings(image_list, cfg_path)
visual_embeds_ = torch.cat([torch.unsqueeze(visual_embeds[i], 0) for i in range(len(visual_embeds))], 0) # in tensor

outputs = vb_model(input_ids=inputs.data.get("input_ids"), visual_embeds=visual_embeds_)
last_hidden_state = outputs.last_hidden_state




















