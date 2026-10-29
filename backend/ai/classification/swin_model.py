"""Swin Transformer classifier with a Dropout + Linear head."""

from __future__ import annotations

from torch import nn
from torchvision import models


def _load_swin_t(pretrained: bool) -> nn.Module:
    if hasattr(models, "swin_t"):
        weights = None
        if pretrained:
            weights_enum = getattr(models, "Swin_T_Weights", None)
            if weights_enum is not None:
                weights = weights_enum.IMAGENET1K_V1
                return models.swin_t(weights=weights)
            return models.swin_t(pretrained=True)
        try:
            return models.swin_t(weights=None)
        except TypeError:
            return models.swin_t(pretrained=False)

    raise RuntimeError(
        "torchvision.models.swin_t is not available in the installed torchvision version."
    )


def _replace_classifier(model: nn.Module, num_classes: int, dropout: float) -> nn.Module:
    if not hasattr(model, "head"):
        raise AttributeError("Unexpected Swin architecture: missing 'head' attribute.")

    head = model.head
    if isinstance(head, nn.Linear):
        in_features = head.in_features
    elif isinstance(head, nn.Sequential) and isinstance(head[-1], nn.Linear):
        in_features = head[-1].in_features
    else:
        raise AttributeError(f"Unsupported Swin head type: {type(head)}")

    model.head = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    return model


def build_swin_classifier(num_classes: int, dropout: float = 0.2, pretrained: bool = True) -> nn.Module:
    if num_classes < 1:
        raise ValueError("num_classes must be obtained from class_names.json and be >= 1")
    model = _load_swin_t(pretrained=pretrained)
    return _replace_classifier(model, num_classes=num_classes, dropout=dropout)


def freeze_backbone(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("head")


def unfreeze_last_two_stages(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if name.startswith("head") or name.startswith("norm"):
            param.requires_grad = True

    if not hasattr(model, "features"):
        return

    stage_indices = [
        index
        for index, module in enumerate(model.features)
        if isinstance(module, nn.Sequential)
    ]
    if len(stage_indices) < 2:
        for param in model.features.parameters():
            param.requires_grad = True
        return

    last_two = set(stage_indices[-2:])
    for index in list(last_two):
        if index > 0:
            last_two.add(index - 1)

    for index, module in enumerate(model.features):
        if index in last_two:
            for param in module.parameters():
                param.requires_grad = True


def trainable_parameter_groups(model: nn.Module, backbone_lr: float, head_lr: float) -> list[dict]:
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("head"):
            head_params.append(param)
        else:
            backbone_params.append(param)

    groups = []
    if backbone_params:
        groups.append({"params": backbone_params, "lr": backbone_lr})
    if head_params:
        groups.append({"params": head_params, "lr": head_lr})
    if not groups:
        raise RuntimeError("No trainable parameters found.")
    return groups


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)
