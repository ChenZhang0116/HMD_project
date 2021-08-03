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
N_image = 4
image_list = [cv2.cvtColor(cv2.imread(str(data_dir / train_samples_frame.loc[i, "img"])), cv2.COLOR_RGB2BGR)
              for i in range(N_image)]
text_list = [train_samples_frame.loc[i, "text"] for i in range(N_image)]
text = train_samples_frame.loc[107, "text"]

model_dir = Path.cwd().parent / "downloaded_models"

from transformers import BertTokenizer, VisualBertModel, VisualBertConfig
from get_visual_embeddings import get_visual_embeddings_cuda

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

encoded = tokenizer.encode_plus(
    text=text,
    add_special_tokens=True,  # Add [CLS] and [SEP]
    max_length=64,  # maximum length of a sentence
    pad_to_max_length=True,  # Add [PAD]s
    truncation=True,
    return_attention_mask=True,  # Generate the attention mask
    #return_tensors='pt',  # ask the function to return PyTorch tensors
)
encoded2 = tokenizer.encode_plus(
    text=text,
    add_special_tokens=True,  # Add [CLS] and [SEP]
    max_length=64,  # maximum length of a sentence
    pad_to_max_length=True,  # Add [PAD]s
    truncation=True,
    return_attention_mask=True,  # Generate the attention mask
    return_tensors='pt',  # ask the function to return PyTorch tensors
)
encoded_batch = tokenizer(text_list, max_length=70, padding=True, add_special_tokens=True, pad_to_max_length=True,
                          truncation=True, return_tensors="pt")
inputs = tokenizer(text_list, padding=True, truncation=True, return_tensors="pt")



vb_conf = VisualBertConfig.from_pretrained('uclanlp/visualbert-vqa-coco-pre')

vb_model = VisualBertModel.from_pretrained(model_dir / "uclanlp-visualbert-nlvr2-coco-pre")
#vb_model = VisualBertModel.from_pretrained('uclanlp/visualbert-vqa-coco-pre', config=vb_conf)
dummy_tensor = torch.zeros(4, 3, 224, 224, device='cuda')

cfg_path = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
visual_embeds = get_visual_embeddings_cuda(dummy_tensor, cfg_path, N_image=4)  # in cuda

text = inputs.data.get("input_ids")
text = text.to('cuda')
vb_model = vb_model.to(torch.device("cuda"))
outputs = vb_model(input_ids=text, visual_embeds=visual_embeds)
last_hidden_state = outputs.last_hidden_state  # Tensor(N, 116, 768)

