import torch
from pathlib import Path
from get_visual_embeddings import get_visual_embeddings_cuda
from transformers import VisualBertModel

"""this architecture realizes mid-level fusion: 
In our LanguageAndVisionConcat architecture, we'll run our image data 
mode through an image model, taking the last set of feature representations as output, then the same for our languge 
mode. Then we'll concatenate these feature representations and treat them as a new feature vector, and send it 
through a final fully connected layer for classification. """


class LanguageAndVisionConcat(torch.nn.Module):
    def __init__(
            self,
            num_classes,
            loss_fn,

            # language_module, # treat the language&vision modules as parameters of this mid-level fusion model
            # vision_module,

            language_feature_dim,
            vision_feature_dim,
            fusion_output_size,
            dropout_p,

    ):
        super(LanguageAndVisionConcat, self).__init__()
        # self.language_module = language_module
        # self.vision_module = vision_module
        self.fusion = torch.nn.Linear(
            in_features=400,
            out_features=fusion_output_size
        )

        self.fc = torch.nn.Linear(
            in_features=153600,
            out_features=num_classes
        )
        self.loss_fn = loss_fn
        self.dropout = torch.nn.Dropout(dropout_p)

        self.vbModel_dir = Path.cwd().parent / "downloaded_models"
        self.vbModel = VisualBertModel.from_pretrained(self.vbModel_dir / "uclanlp-visualbert-nlvr2-coco-pre")

    def forward(self, text, image, label=None):
        # text_features = torch.nn.functional.relu(self.language_module(text))

        # print("\ntext - " + str(text.shape) + ", " + str(text.dtype))

        image = image.to(torch.device('cuda'))  # add this during test time !!!!

        cfg_path = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
        visual_embedds = get_visual_embeddings_cuda(image, cfg_path, N_image=image.shape[0])

        # print("visual_embeds decive:" + visual_embedds.device.type)

        # image_features = torch.nn.functional.relu(self.vision_module(image))
        # combined = torch.cat([text, image_features], dim=1)

        text_ = torch.tensor(text).to('cuda').long()
        # print("text_ device: " + text_.device.type)

        outputs = self.vbModel(input_ids=text_, visual_embeds=visual_embedds)
        last_hidden_state = outputs.last_hidden_state  # (B,200,768)
        last_hidden_state_flat = torch.flatten(last_hidden_state, 1, -1)

        # print("combined size = " + str(combined.shape))
        # print("vb model output size = " + str(last_hidden_state_flat.shape))

        # fused = self.fusion(combined)
        # fused = torch.nn.functional.relu(fused)
        # fused = self.dropout(fused)

        logits = self.fc(last_hidden_state_flat)
        pred = torch.nn.functional.softmax(logits)
        loss = (
            self.loss_fn(pred, label)
            if label is not None else label
        )
        return (pred, loss)
