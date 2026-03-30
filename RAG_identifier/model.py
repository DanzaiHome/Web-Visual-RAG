import torch
import torch.nn as nn

from transformers import SiglipModel, AutoTokenizer


class CLIPRouter(nn.Module):

    def __init__(
        self,
        model_name="google/siglip-so400m-patch14-384",
        hidden_dim=256,
        freeze_clip=True
    ):
        super().__init__()

        # load SigLIP
        self.siglip = SiglipModel.from_pretrained(model_name)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        embed_dim = self.siglip.config.vision_config.hidden_size # 1024

        # freeze encoder
        if freeze_clip:
            for p in self.siglip.parameters():
                p.requires_grad = False

        # classifier
        input_dim = embed_dim * 3

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, images, questions):

        device = images.device

        # image features
        image_outputs = self.siglip.vision_model(
            pixel_values=images
        )

        image_features = image_outputs.pooler_output

        # text features
        text_inputs = self.tokenizer(
            questions,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(device)

        text_outputs = self.siglip.text_model(
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs.get("attention_mask", None)
        )
        
        text_features = text_outputs.pooler_output
        
        # normalize
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # feature fusion
        diff_feature = text_features - image_features

        features = torch.cat(
            [image_features, text_features, diff_feature],
            dim=1
        ).float()

        logits = self.classifier(features)

        return logits.squeeze(-1)