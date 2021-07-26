import torch

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

            language_module, # treat the language&vision modules as parameters of this mid-level fusion model
            vision_module,

            language_feature_dim,
            vision_feature_dim,
            fusion_output_size,
            dropout_p,

    ):
        super(LanguageAndVisionConcat, self).__init__()
        self.language_module = language_module
        self.vision_module = vision_module
        self.fusion = torch.nn.Linear(
            in_features=(language_feature_dim + vision_feature_dim),
            out_features=fusion_output_size
        )
        self.fc = torch.nn.Linear(
            in_features=fusion_output_size,
            out_features=num_classes
        )
        self.loss_fn = loss_fn
        self.dropout = torch.nn.Dropout(dropout_p)

    def forward(self, text, image, label=None):
        text_features = torch.nn.functional.relu(self.language_module(text))
        image_features = torch.nn.functional.relu(self.vision_module(image))
        combined = torch.cat([text_features, image_features], dim=1)
        fused = self.dropout(torch.nn.functional.relu(self.fusion(combined)))
        logits = self.fc(fused)
        pred = torch.nn.functional.softmax(logits)
        loss = (
            self.loss_fn(pred, label)
            if label is not None else label
        )
        return (pred, loss)
