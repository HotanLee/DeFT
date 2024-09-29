import math
import torch
import torch.nn as nn
from torch.nn import functional as F

from .peft_modelus import *
from .text_encoder import *

class ViT_Tuner(nn.Module):
    def __init__(self, cfg, clip_model, text_features):
        super().__init__()
        n_layers = clip_model.visual.transformer.layers
        emb_dim = clip_model.visual.transformer.width
        seq_len = clip_model.visual.positional_embedding.shape[0]
        patch_size = clip_model.visual.conv1.kernel_size
        dtype = clip_model.dtype

        use_finetune = cfg.finetune
        use_bias_tuning = cfg.bias_tuning
        use_vpt_shallow = cfg.vpt_shallow
        use_vpt_deep = cfg.vpt_deep
        use_adapter = cfg.adapter
        use_lora = cfg.lora
        use_ssf = cfg.ssf
        vpt_len = cfg.vpt_len
        adapter_dim = cfg.adapter_dim
        lora_dim = cfg.lora_dim
        partial = cfg.partial

        if partial is None:
            partial = n_layers
        else:
            partial = int(partial)

        blocks = clip_model.visual.transformer.resblocks

        if use_finetune:
            finetune_list = blocks
        else:
            finetune_list = None

        if use_bias_tuning:
            bias_list = nn.ParameterList([
                param for name, param in clip_model.visual.named_parameters()
                if name.endswith("bias")
            ])
        else:
            bias_list = None

        assert int(use_vpt_shallow) + int(use_vpt_deep) < 2
        if use_vpt_shallow:
            vpt_list = nn.ModuleList([
                VPT(vpt_len=vpt_len, seq_len=seq_len, patch_size=patch_size, emb_dim=emb_dim, dtype=dtype),
                *[None] * (n_layers - 1)
            ])
        elif use_vpt_deep:
            vpt_list = nn.ModuleList([
                *[None] * (n_layers - partial),
                *[VPT(vpt_len=vpt_len, seq_len=seq_len, patch_size=patch_size, emb_dim=emb_dim, dtype=dtype) for _ in range(partial)]
            ])
        else:
            vpt_list = [None] * n_layers
        
        if use_adapter:
            adapter_list = nn.ModuleList([
                *[None] * (n_layers - partial),
                *[Adapter(in_dim=emb_dim, bottle_dim=adapter_dim, dtype=dtype) for _ in range(partial)]
            ])
        else:
            adapter_list = [None] * n_layers

        if use_lora:
            lora_list = nn.ModuleList([
                *[None] * (n_layers - partial),
                *[nn.ModuleDict({
                    "q": LoRA(in_dim=emb_dim, bottle_dim=lora_dim, dtype=dtype),
                    "v": LoRA(in_dim=emb_dim, bottle_dim=lora_dim, dtype=dtype),
                }) for _ in range(partial)]
            ])
        else:
            lora_list = [None] * n_layers

        if use_ssf:
            _block_0 = clip_model.visual.transformer.resblocks[0]
            ssf_list = nn.ModuleList([
                *[None] * (n_layers - partial),
                *[nn.ModuleDict({
                    "attn_in": SSF(_block_0.attn.in_proj_bias.shape[0], dtype=dtype),
                    "attn_out": SSF(_block_0.attn.out_proj.bias.shape[0], dtype=dtype),
                    "mlp_in": SSF(_block_0.mlp[0].bias.shape[0], dtype=dtype),
                    "mlp_out": SSF(_block_0.mlp[2].bias.shape[0], dtype=dtype),
                }) for _ in range(partial)]
            ])
        else:
            ssf_list = [None] * n_layers

        visual_proj = clip_model.visual.proj.data
        head = nn.Linear(visual_proj.shape[0], cfg.num_class).to(clip_model.dtype)
        if text_features is not None:
            head.weight.data = text_features.data @ visual_proj.data.t()  # (100, 512) @ (512, 768) = (100, 768)

        # To be optimized
        self.finetune_list = finetune_list
        self.bias_list = bias_list
        self.vpt_list = vpt_list
        self.adapter_list = adapter_list
        self.lora_list = lora_list
        self.ssf_list = ssf_list
        self.head = head


class CLIP_ViT(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.conv1 = clip_model.visual.conv1
        self.class_embedding = clip_model.visual.class_embedding
        self.positional_embedding = clip_model.visual.positional_embedding
        self.ln_pre = clip_model.visual.ln_pre
        self.transformer = clip_model.visual.transformer
        self.ln_post = clip_model.visual.ln_post
        self.proj = clip_model.visual.proj
        self.dtype = clip_model.dtype

    def forward(self, x, tuner=None):
        x = x.to(self.dtype)
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        _bsz = x.shape[0]
        _seq_len = x.shape[1]
        _emb_dim = x.shape[2]

        n_layers = self.transformer.layers

        for i in range(n_layers):
            block = self.transformer.resblocks[i]

            if tuner is not None:
                vpt = tuner.vpt_list[i]
                adapter = tuner.adapter_list[i]
                lora = tuner.lora_list[i]
                ssf = tuner.ssf_list[i]
            else:
                vpt = adapter = lora = ssf = None

            if vpt is not None:
                x = vpt(x)

            _seq_len_after_vpt = x.shape[1]

            x = x.permute(1, 0, 2)  # NLD -> LND

            _attn = block.attn
            _ln_1 = block.ln_1
            _mlp = block.mlp
            _ln_2 = block.ln_2

            _attn_in_proj_weight = _attn.in_proj_weight
            _attn_in_proj_bias = _attn.in_proj_bias
            _attn_out_proj_weight = _attn.out_proj.weight
            _attn_out_proj_bias = _attn.out_proj.bias
            _mlp_in_proj = _mlp[0]
            _mlp_gelu = _mlp[1]
            _mlp_out_proj = _mlp[2]

            _num_heads = _attn.num_heads
            _head_dim = _emb_dim // _num_heads

            ###############################
            ## Multi-Head Self-Attention ##
            ###############################
            residual = x  # deep copy

            x = _ln_1(x)

            qkv = F.linear(x, _attn_in_proj_weight, _attn_in_proj_bias)
            if ssf is not None:
                qkv = ssf["attn_in"](qkv)
            q, k, v = qkv.chunk(3, dim=-1)

            if lora is not None:
                q = q + lora["q"](x)
                v = v + lora["v"](x)

            q = q.contiguous().view(q.shape[0], q.shape[1] * _num_heads, _head_dim).transpose(0, 1)
            k = k.contiguous().view(k.shape[0], k.shape[1] * _num_heads, _head_dim).transpose(0, 1)
            v = v.contiguous().view(v.shape[0], v.shape[1] * _num_heads, _head_dim).transpose(0, 1)
            
            q = q / math.sqrt(_head_dim)
            attn = torch.bmm(q, k.transpose(-2, -1))
            attn = F.softmax(attn, dim=-1)
            x = torch.bmm(attn, v)
            x = x.transpose(0, 1).contiguous().view(-1, _emb_dim)
            
            x = F.linear(x, _attn_out_proj_weight, _attn_out_proj_bias)
            if ssf is not None:
                x = ssf["attn_out"](x)
            x = x.view(_seq_len_after_vpt, _bsz, _emb_dim)

            x = residual + x

            ##########################
            ## Feed-Forward Network ##
            ##########################
            residual = x  # deep copy

            x = _ln_2(x)

            x = _mlp_in_proj(x)
            if ssf is not None:
                x = ssf["mlp_in"](x)
            x = _mlp_gelu(x)
            x = _mlp_out_proj(x)
            if ssf is not None:
                x = ssf["mlp_out"](x)
            
            if adapter is not None:
                x = adapter(x)
            
            x = residual + x
            
            x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])
        return x


class Model(nn.Module):
    def __init__(self, cfg, clip_model, text_features=None):
        super().__init__()
        # image encoder
        self.image_encoder = CLIP_ViT(clip_model)
        self.tuner = ViT_Tuner(cfg, clip_model, text_features=text_features)
        # text encoder
        self.prompt_learner = PromptLearner(cfg, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.text_encoder = TextEncoder(clip_model)

        self.logit_scale = clip_model.logit_scale

    def forward(self, image, return_sim=False, return_neg=False):
        # image features
        feats = self.image_encoder(image, self.tuner)
        image_features = feats @ self.image_encoder.proj

        # dual text features
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features, neg_text_features = self.text_encoder(prompts, tokenized_prompts)

        # Normalization
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        neg_text_features = neg_text_features / neg_text_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()

        if return_neg:  # dual prompt learning
            logits_pos = logit_scale * image_features @ text_features.t()
            logits_neg = logit_scale * image_features.detach() @ neg_text_features.t()
            return logits_pos, logits_neg
        elif return_sim:  # textual classifier
            logits_pos = logit_scale * image_features @ text_features.t()
            return logits_pos
        else:  # linear classifier
            head = self.tuner.head
            logits = head(feats)
            return logits
