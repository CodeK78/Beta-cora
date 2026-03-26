"""
3D Residual U-Net encoder-decoder for synthesis-driven self-supervised pretraining,
with downstream task heads for classification and MACE risk stratification.
"""

import torch
from torch import nn
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
from transformers import AutoModel, AutoTokenizer


# =============================================================================
# Core Architecture
# =============================================================================

def build_unet(
    num_input_channels: int = 4,
    num_output_channels: int = 1,
    deep_supervision: bool = False,
) -> nn.Module:
    """
    Build a 3D Residual U-Net (4-stage encoder-decoder).

    Args:
        num_input_channels: Number of input channels (4 for multi-window CCTA).
        num_output_channels: Number of segmentation output channels.
        deep_supervision: Whether to enable deep supervision.

    Returns:
        A ResidualEncoderUNet instance.
    """
    n_stages = 4
    model = ResidualEncoderUNet(
        input_channels=num_input_channels,
        n_stages=n_stages,
        features_per_stage=[32, 64, 128, 256],
        conv_op=nn.Conv3d,
        kernel_sizes=[[3, 3, 3]] * n_stages,
        strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
        n_blocks_per_stage=[1, 3, 4, 4],
        num_classes=num_output_channels,
        n_conv_per_stage_decoder=[1, 1, 1],
        conv_bias=True,
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={"eps": 1e-5, "affine": True},
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
        deep_supervision=deep_supervision,
    )
    return model


# =============================================================================
# Pooling Layers
# =============================================================================

class AdaptiveConcatPool3d(nn.Module):
    """Concatenation of adaptive average and max pooling (doubles feature dim)."""

    def __init__(self):
        super().__init__()
        self.ap = nn.AdaptiveAvgPool3d(1)
        self.mp = nn.AdaptiveMaxPool3d(1)

    def forward(self, x):
        return torch.cat([self.ap(x), self.mp(x)], dim=1)


# =============================================================================
# Pretraining Model: Abnormality Detection (Segmentation)
# =============================================================================

class CORAPretrainModel(nn.Module):
    """
    CORA pretraining model for synthesis-driven self-supervised learning.

    Performs abnormality segmentation using the full U-Net encoder-decoder.
    The encoder learns pathology-centric representations; the decoder produces
    voxel-level abnormality response maps.
    """

    def __init__(self, num_input_channels=4, num_output_channels=1):
        super().__init__()
        self.unet = build_unet(num_input_channels, num_output_channels)
        self.bottleneck_channels = 256
        self.adaptive_pool = AdaptiveConcatPool3d()
        self.cls_head = nn.Sequential(
            nn.Linear(self.bottleneck_channels * 2, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        # Encoder: extract multi-scale features
        skips = self.unet.encoder(x)
        bottleneck = skips[-1]

        # Classification branch (auxiliary)
        global_feat = self.adaptive_pool(bottleneck).view(x.size(0), -1)
        logits_cls = self.cls_head(global_feat)

        # Segmentation branch: decode with skip connections
        logits_seg = self.unet.decoder(skips)

        return logits_cls, logits_seg


# =============================================================================
# Downstream: Classification (Plaque Characterization)
# =============================================================================

class CORAClassifier(nn.Module):
    """
    Downstream classifier using the pretrained CORA encoder.

    Produces two outputs for multi-label plaque classification:
    - Head 1: single logit (e.g., MACE binary prediction)
    - Head 2: two logits (calcified / non-calcified plaque)
    """

    def __init__(self, num_input_channels=4):
        super().__init__()

        # Extract encoder from full U-Net
        full_unet = build_unet(num_input_channels, num_output_channels=1)
        self.encoder = full_unet.encoder
        del full_unet.decoder, full_unet

        self.bottleneck_channels = 256
        self.adaptive_pool = AdaptiveConcatPool3d()

        self.cls_head1 = nn.Linear(self.bottleneck_channels * 2, 1)
        self.cls_head2 = nn.Linear(self.bottleneck_channels * 2, 2)

    def forward(self, x):
        skips = self.encoder(x)
        bottleneck = skips[-1]

        global_feat = self.adaptive_pool(bottleneck).view(x.size(0), -1)
        logits1 = self.cls_head1(global_feat)
        logits2 = self.cls_head2(global_feat)

        return logits1, logits2


# =============================================================================
# Downstream: Multimodal MACE Prediction (Image + LLM Text Encoder)
# =============================================================================

class CORAMultimodalMACE(nn.Module):
    """
    Multimodal MACE risk stratification model.

    Fuses CORA image encoder features with clinical text embeddings from a
    frozen Qwen language model for 30-day MACE prediction.

    Args:
        num_input_channels: Number of input image channels.
        qwen_model_path: Path or HuggingFace ID of the Qwen model.
    """

    def __init__(self, num_input_channels=4, qwen_model_path="Qwen/Qwen2.5-7B"):
        super().__init__()

        # Image encoder
        full_unet = build_unet(num_input_channels, num_output_channels=1)
        self.encoder = full_unet.encoder
        del full_unet.decoder, full_unet

        self.bottleneck_channels = 256
        self.adaptive_pool = AdaptiveConcatPool3d()

        # Frozen Qwen text encoder
        self.tokenizer = AutoTokenizer.from_pretrained(
            qwen_model_path, trust_remote_code=True
        )
        self.text_model = AutoModel.from_pretrained(
            qwen_model_path, trust_remote_code=True
        )
        for param in self.text_model.parameters():
            param.requires_grad = False
        self.text_model.eval()

        self.text_embed_dim = self.text_model.config.hidden_size

        # Text projection: map LLM hidden dim -> 256
        self.text_projection = nn.Sequential(
            nn.Linear(self.text_embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )

        # Fusion classifier: image (512) + text (256) -> binary MACE prediction
        self.combined_dim = (self.bottleneck_channels * 2) + 256
        self.cls_head = nn.Sequential(
            nn.Linear(self.combined_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )

    def forward(self, x_img, raw_texts):
        """
        Args:
            x_img: Image tensor of shape [B, C, D, H, W].
            raw_texts: List of clinical text strings, length B.

        Returns:
            logits: MACE prediction logits of shape [B, 1].
        """
        # Image features
        skips = self.encoder(x_img)
        img_feat = self.adaptive_pool(skips[-1]).view(x_img.size(0), -1)

        # Text features (frozen)
        inputs = self.tokenizer(
            raw_texts, return_tensors="pt",
            padding=True, truncation=True, max_length=512,
        )
        inputs = {k: v.to(x_img.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.text_model(**inputs)
            text_embeds = outputs.last_hidden_state.mean(dim=1)

        text_feat = self.text_projection(text_embeds)

        # Fusion and classification
        combined = torch.cat([img_feat, text_feat], dim=1)
        logits = self.cls_head(combined)

        return logits
