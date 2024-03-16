import os.path as osp

import torch
import torch.nn as nn
import copy
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from visual_augment import VisualAugment, ParamVisualAug

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    design_details = {"vp_length": cfg.TRAINER.COOP.VP_LENGTH}

    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        single_text_features = x
        x, hidden_text_features = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), 7] @ self.text_projection
        return x, single_text_features, hidden_text_features

class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            ctx_init = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
            ctx_init = ctx_init.replace(" {}.", "")
            ctx_init = ctx_init.replace("_", " ")
            prompt_n_ctx = len(ctx_init.split(" "))

            assert n_ctx >= prompt_n_ctx, f"#tokens ({n_ctx}) should larger equal than #initial prompt tokens ({prompt_n_ctx}, {ctx_init})"

            prompt = clip.tokenize(ctx_init)

            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)

            ctx_vectors = torch.zeros(n_ctx, ctx_dim, dtype=dtype)

            ctx_vectors[n_ctx - prompt_n_ctx:, :] = embedding[0, 1:1 +
                                                              prompt_n_ctx, :]
            prompt_prefix = " ".join(["X"] * (n_ctx - prompt_n_ctx))
            prompt_prefix = f"{prompt_prefix} {ctx_init}"
        else:
            # random initialization
            if cfg.TRAINER.COOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)

        cls_init = [name.replace("_", " ") for name in classnames]
        # print(cls_init)
        cls_token = torch.cat([clip.tokenize(name) for name in cls_init])

        with torch.no_grad():
            cls_embedding = clip_model.token_embedding(cls_token).type(dtype)
        # print(cls_embedding.shape)
        cls_vectors = torch.zeros(n_cls, 1, ctx_dim, dtype=dtype)
        cls_vectors[:, :, :] = cls_embedding[:, 1:2, :]
        
        self.cls = nn.Parameter(cls_vectors)  # to be optimized
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        # print(prompts)
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(
                dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS

        ending_note = "."
        ending_token = clip.tokenize(ending_note)
        # print(ending_token)
        suffix_token = ending_token[:, 1: -(n_ctx+1)]
        # print(suffix_token.shape)

        with torch.no_grad():
            suffix_embedding = clip_model.token_embedding(suffix_token).type(
                dtype)

        suffix_embedding = suffix_embedding.expand(n_cls, -1, -1)

        self.register_buffer("token_suffix", suffix_embedding)  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx
        cls_token = self.cls
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        content = torch.cat((ctx, cls_token), dim=1)

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    content,  # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        else:
            raise ValueError
            
        return prompts, cls_token


class MIModule(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        n_vp = cfg.TRAINER.COOP.VP_LENGTH
        self.prompts_depth = cfg.TRAINER.COOP.PROMPT_DEPTH
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        self.n_ctx = n_ctx
        self.single_proj_dim = nn.Linear(ctx_dim, 768)
        self.single_proj_dim.half()
        self.proj_dim = _get_clones(self.single_proj_dim, self.prompts_depth - 1)
        self.relu = nn.ReLU(inplace=True)
        self.single_proj_cls = nn.Linear(n_cls, n_vp)
        self.single_proj_cls.half()
        self.proj_cls = _get_clones(self.single_proj_cls, self.prompts_depth - 1)
        self.n_vp = n_vp

    def forward(self, text_prompts, hidden_text_features):
        self.text_prompts = text_prompts
        self.hidden_text_features = hidden_text_features
        self.cls_token = self.text_prompts[1+self.n_ctx :1+self.n_ctx +1,:,:]
        self.dense_cls_token = self.hidden_text_features[:,1+self.n_ctx :1+self.n_ctx +1,:,:]

        initial_vp = torch.zeros(self.n_vp, 768).half().to(self.cls_token.device)

        dense_vp = []
        for index, (layer_dim, layer_cls) in enumerate(zip(self.proj_dim, self.proj_cls)):
            dense_cls_token_layer = self.dense_cls_token[index].mean(dim=0)
            vp1 = layer_dim(dense_cls_token_layer) 
            vp1 = self.relu(vp1)
            vp1 = vp1.permute(1, 0)
            vp2 = layer_cls(vp1)
            vp2 = vp2.permute(1, 0)
            dense_vp.append(vp2)
        
        return initial_vp, dense_vp
                

CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}.",
    "OxfordFlowers": "a photo of a {}.",
    "FGVCAircraft": "a photo of a {}.",
    "DescribableTextures": "a photo of a {}.",
    "EuroSAT": "a photo of a {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of a {}.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a {}.",
    "RESISC45": "a photo of a {}.",
    "PatternNet": "a photo of a {}.",
    "RSICD": "a photo of a {}.",
    "MLRSNet": "a photo of a {}.",
    "BTMRI": "a photo of a {}.",
    "CHMNIST": "a photo of a {}.",
    "CCBTM": "a photo of a {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
}


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.mimodule = MIModule(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        # self.augment = 'color_crop_cutout_flip_scale_rotate'
        # self.param = ParamVisualAug()

    def forward(self, image):
        prompts, cls_token = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features, single_text_features, hidden_text_features = self.text_encoder(prompts, tokenized_prompts)
        vp, dense_vp = self.mimodule(single_text_features, hidden_text_features)

        image_features = self.image_encoder(image.type(self.dtype), vp, dense_vp)

        image_features = image_features / image_features.norm(dim=-1,
                                                              keepdim=True)

        text_features = text_features / text_features.norm(dim=-1,
                                                           keepdim=True)
        
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits, image_features


@TRAINER_REGISTRY.register()
class MITLOp(TrainerX):

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        print(classnames)
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name and "mimodule" not in name:
                param.requires_grad_(False)

        enabled = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.append(name)
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model,
                                    cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model,
                            self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(
                f"Multiple GPUs detected (n_gpus={device_count}), use all of them!"
            )
            self.model = nn.DataParallel(self.model)

    def image_augment(self, image):
        # print(self.cfg.TRAINER.COOP.AUGMENT)
        self.augment = self.cfg.TRAINER.COOP.AUGMENT
        self.param = ParamVisualAug()
        image = VisualAugment(image, self.augment, seed=-1, param=self.param)
        return image

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        # print(image.shape)
        # image = VisualAugment(image, self.augment, seed=-1, param=self.param)
        image_a = self.image_augment(image)

        prec = self.cfg.TRAINER.COOP.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output, img_features = self.model(image)
            output_a, img_features_a = self.model(image_a)

            loss_1 = F.cross_entropy(output, label) 
            loss_consis = 1 - F.cosine_similarity(img_features,img_features_a)
            loss_consis = loss_consis.mean()

            loss = loss_1+10*loss_consis

            self.model_backward_and_update(loss)


        loss_summary = {
            "loss_1": loss_1.item(),
            "loss_consis": loss_consis.item(),
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print(
                "Note that load_model() is skipped as no pretrained model is given"
            )
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError(
                    'Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} "
                  'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
