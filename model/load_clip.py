import torch
from clip import clip
from .CLIP import *

def load_clip(cfg):
    print(f"Loading CLIP (backbone: {cfg.backbone})")
    backbone_name = cfg.backbone
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url, './model/clip/weights')

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    clip_model = clip.build_model(state_dict or model.state_dict())
    clip_model.cuda()

    assert cfg.prec in ["fp16", "fp32", "amp"]
    if cfg.prec == "fp32" or cfg.prec == "amp":
        # CLIP's default precision is fp16
        clip_model.float()

    model = Model(cfg, clip_model)
    tuner = model.tuner

    # Turning off gradients in the model"
    for name, param in model.named_parameters():
        param.requires_grad_(False)
    # Turning on gradients in the tuner
    for name, param in tuner.named_parameters():
        param.requires_grad_(True)

    # print total parameters and tuned parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total params: {total_params}')
    tuned_params = sum(p.numel() for p in tuner.parameters())
    print(f'Image Tuned params: {tuned_params}')
    head_params = sum(p.numel() for p in tuner.head.parameters())
    tuned_params_without_head = tuned_params - head_params
    print(f'Tuned params (w/o head): {tuned_params_without_head}')

    optimizer = torch.optim.SGD(tuner.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, momentum=cfg.momentum)

    return model, optimizer


def load_deft(cfg):
    print(f"Loading CLIP (backbone: {cfg.backbone})")
    backbone_name = cfg.backbone
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url, './model/clip/weights')

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    clip_model = clip.build_model(state_dict or model.state_dict())
    clip_model.cuda()

    assert cfg.prec in ["fp16", "fp32", "amp"]
    if cfg.prec == "fp32" or cfg.prec == "amp":
        # CLIP's default precision is fp16
        clip_model.float()
    
    model = Model(cfg, clip_model)
    tuner = model.tuner
    prompt_learner = model.prompt_learner

    # Turning off gradients in the model"
    for name, param in model.named_parameters():
        param.requires_grad_(False)
    # Turning on gradients in the tuner
    for name, param in tuner.named_parameters():
        param.requires_grad_(True)
    for name, param in prompt_learner.named_parameters():
        param.requires_grad_(True)

    # print total parameters and tuned parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total params: {total_params}')
    prompt_tune = sum(p.numel() for p in prompt_learner.parameters())
    print(f'Text Prompt params: {prompt_tune}')
    tuned_params = sum(p.numel() for p in tuner.parameters())
    head_params = sum(p.numel() for p in tuner.head.parameters())
    tuned_params_without_head = tuned_params - head_params
    print(f'Image Tuned params: {tuned_params_without_head}')

    parameters_to_optim = [
        {"params": tuner.parameters()},
        {"params": prompt_learner.parameters()}
    ]
    optimizer = torch.optim.SGD(parameters_to_optim, lr=cfg.lr, weight_decay=cfg.weight_decay, momentum=cfg.momentum)

    return model, optimizer