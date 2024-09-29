import torch
import torch.nn as nn
from torch.nn import functional as F

from .clip import clip
from .clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward_one_prompt(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x
    
    def forward(self, prompts, tokenized_prompts):
        pos_txt_features = self.forward_one_prompt(prompts[0], tokenized_prompts[0])
        neg_txt_features = self.forward_one_prompt(prompts[1], tokenized_prompts[1])

        return pos_txt_features, neg_txt_features


class PromptLearner(nn.Module):
    def __init__(self, cfg, clip_model):
        super().__init__()
        classnames = cfg.class_names
        n_cls = len(classnames)
        n_ctx = cfg.N_CTX
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        # random initialization
        # print("Initializing a generic context for POS/NEG prompt")
        pos_ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
        neg_ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(pos_ctx_vectors, std=0.02)
        nn.init.normal_(neg_ctx_vectors, std=0.02)
        pos_prompt_prefix = " ".join(["P"] * n_ctx)
        neg_prompt_prefix = " ".join(["N"] * n_ctx)

        # print(f'Initial positive context: "{pos_prompt_prefix}"')
        # print(f'Initial negative context: "{neg_prompt_prefix}"')
        # print(f"Number of context words (tokens): {n_ctx}")

        self.pos_ctx = nn.Parameter(pos_ctx_vectors)  # to be optimized
        self.neg_ctx = nn.Parameter(neg_ctx_vectors)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]

        pos_prompts = [pos_prompt_prefix + " " + name + "." for name in classnames]
        neg_prompts = [neg_prompt_prefix + " " + name + "." for name in classnames]

        pos_tokenized_prompts = torch.cat([clip.tokenize(p) for p in pos_prompts]).cuda()
        neg_tokenized_prompts = torch.cat([clip.tokenize(p) for p in neg_prompts]).cuda()
        with torch.no_grad():
            embedding = clip_model.token_embedding(pos_tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = [pos_tokenized_prompts, neg_tokenized_prompts]  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.CLASS_TOKEN_POSITION

    def forward_one_prompt(self, ctx):
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts
    
    def forward(self):
        pos_prompts = self.forward_one_prompt(self.pos_ctx)
        neg_prompts = self.forward_one_prompt(self.neg_ctx)

        return [pos_prompts, neg_prompts]