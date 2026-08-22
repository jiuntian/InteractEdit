import glob
import itertools
import json
import math
import os
from packaging import version
from typing import List, Literal, Optional, Tuple

from PIL import Image
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, set_seed
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    StableDiffusionXLPipeline,
)
from diffusers.models.attention_processor import (
    Attention,
)
from diffusers.optimization import get_scheduler
from diffusers.schedulers import DDPMScheduler, DDIMScheduler, KarrasDiffusionSchedulers
from diffusers.training_utils import cast_training_params
from diffusers.utils import (
    check_min_version,
    convert_state_dict_to_diffusers,
)
from diffusers.utils.import_utils import is_xformers_available
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from safetensors import safe_open
import safetensors
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
    PretrainedConfig,
)

from custom_lora import CustomLinear
from segment import GroundedSAM, annotate, DetectionResult
from misc import DreamBoothDataset
from ptp_utils import AttentionStore

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.30.0")

logger = get_logger(__name__)

def collate_fn(examples, resolution):
    pixel_values = [example["instance_images"] for example in examples]
    prompts = [example["instance_prompt"] for example in examples]
    masks = [example["instance_masks"] for example in examples]
    token_ids = [example["token_ids"] for example in examples]
    
    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    masks = torch.stack(masks)
    token_ids = torch.stack(token_ids)
    
    batch = {
        "pixel_values": pixel_values,
        "prompts": prompts,
        "instance_masks": masks,
        "token_ids": token_ids,
        "original_sizes": [(resolution, resolution) for _ in examples], # (image.height, image.width)
        "crop_top_lefts": [(0, 0) for _ in examples], # assume no crop
    }
    return batch

def get_average_attention(controller):
    average_attention = {
        key: [
            item / controller.cur_step
            for item in controller.attention_store[key]
        ]
        for key in controller.attention_store
    }
    return average_attention

def aggregate_attention(
    res: int, from_where: List[str], is_cross: bool, select: int, train_batch_size: int, controller,
):
    out = []
    attention_maps = get_average_attention(controller)
    num_pixels = res**2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                # print(item.shape)
                # TODO: potential bug here, batch size incorrect
                cross_maps = item.reshape(
                    train_batch_size, -1, res, res, item.shape[-1]
                )[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out

def save_progress(text_encoder, placeholder_token_ids, placeholder_tokens, accelerator, save_path, safe_serialization=True):
    logger.info("Saving embeddings")
    learned_embeds = (
        accelerator.unwrap_model(text_encoder)
        .get_input_embeddings()
        .weight[min(placeholder_token_ids) : max(placeholder_token_ids) + 1]
    )
    assert len(placeholder_tokens) == learned_embeds.shape[0]
    learned_embeds_dict = {}
    for i, token in enumerate(placeholder_tokens):
        learned_embeds_dict[token] = learned_embeds[i].detach().cpu()

    if safe_serialization:
        safetensors.torch.save_file(learned_embeds_dict, save_path, metadata={"format": "pt"})
    else:
        torch.save(learned_embeds_dict, save_path)

def tokenize_prompt(tokenizer, prompt):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids


# Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
def encode_prompt(text_encoders, tokenizers, prompt,
                  text_input_ids_list=None,
                  clip_skip: Optional[int] = None,):
    prompt_embeds_list = []
    token_ids_tokenizer_1 = None

    for i, text_encoder in enumerate(text_encoders):
        if tokenizers is not None:
            tokenizer = tokenizers[i]
            text_input_ids = tokenize_prompt(tokenizer, prompt)
            if token_ids_tokenizer_1 is None:
                token_ids_tokenizer_1 = text_input_ids
        else:
            assert text_input_ids_list is not None
            text_input_ids = text_input_ids_list[i]

        prompt_embeds = text_encoder(
            text_input_ids.to(text_encoder.device), output_hidden_states=True,
        )

        # We are only ALWAYS interested in the pooled output of the final text encoder
        pooled_prompt_embeds = prompt_embeds[0]
        
        if clip_skip is None:
            prompt_embeds = prompt_embeds.hidden_states[-2]
        else:
            # "2" because SDXL always indexes from the penultimate layer.
            prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]
        
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
        prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds, token_ids_tokenizer_1

def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")

class P2PCrossAttnProcessor:
    def __init__(self, controller, place_in_unet):
        super().__init__()
        self.controller = controller
        self.place_in_unet = place_in_unet

    def __call__(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
    ):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size=batch_size)

        query = attn.to_q(hidden_states)

        is_cross = encoder_hidden_states is not None
        encoder_hidden_states = (
            encoder_hidden_states
            if encoder_hidden_states is not None
            else hidden_states
        )
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        # print(f"first: {attention_probs._version}")

        # one line change
        self.controller(attention_probs, is_cross, self.place_in_unet)
        # print(f"last: {attention_probs._version}")

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

class InteractEditPipeline(StableDiffusionXLPipeline):
    def __init__(self,
                 vae: AutoencoderKL,
                 text_encoder: CLIPTextModel,
                 text_encoder_2: CLIPTextModelWithProjection,
                 tokenizer: CLIPTokenizer,
                 tokenizer_2: CLIPTokenizer,
                 unet: UNet2DConditionModel,
                 scheduler: KarrasDiffusionSchedulers,
                 image_encoder: CLIPVisionModelWithProjection = None,
                 feature_extractor: CLIPImageProcessor = None,
                 force_zeros_for_empty_prompt: bool = True,
                 add_watermarker: Optional[bool] = None,):

        super().__init__(vae,
                         text_encoder,
                         text_encoder_2,
                         tokenizer,
                         tokenizer_2,
                         unet,
                         scheduler,
                         image_encoder,
                         feature_extractor,
                         force_zeros_for_empty_prompt,
                         add_watermarker)

        self.loaded = None  # check whether is loaded/trained

    @torch.no_grad()
    def load_textual_inversion(self, ckpt_path):
        # only being called by load_pipeline

        # original text embed map
        embed_1 = self.text_encoder.get_input_embeddings().weight
        embed_2 = self.text_encoder_2.get_input_embeddings().weight

        files = glob.glob(ckpt_path + "/learned_embeds-steps*")
        assert len(
            files) == 1, f'must have only one textual inversion embeddings, having {len(files)}'
        loaded_embed_1 = safe_open(files[0], framework='pt')

        files_2 = glob.glob(ckpt_path + "/learned_embeds_2-steps*")
        assert len(
            files_2) == 1, f'must have only one textual inversion 2 embeddings, having {len(files)}'
        loaded_embed_2 = safe_open(files_2[0], framework='pt')

        added_token_1 = json.load(
            open(os.path.join(ckpt_path, "tokenizer", "added_tokens.json")))
        added_token_2 = json.load(
            open(os.path.join(ckpt_path, "tokenizer_2", "added_tokens.json")))

        # enlarge text embed map
        assert len(
            embed_1) == self.tokenizer.vocab_size, 'tokenizer is not loaded yet'
        assert len(
            embed_2) == self.tokenizer_2.vocab_size, 'tokenizer_2 is not loaded yet'
        self.text_encoder.resize_token_embeddings(
            len(self.tokenizer))
        self.text_encoder_2.resize_token_embeddings(
            len(self.tokenizer_2))
        assert len(embed_1) == len(
            self.tokenizer), 'size of new tokens mismatched'
        assert len(embed_2) == len(
            self.tokenizer_2), 'size of new tokens mismatched'

        # update new entities tokens
        for k in added_token_1:
            token_id = added_token_1[k]
            embed_1[token_id] = loaded_embed_1.get_tensor(k)

        for k in added_token_2:
            token_id = added_token_2[k]
            embed_2[token_id] = loaded_embed_2.get_tensor(k)

    @classmethod
    def load_trained_pipeline(cls, base_model, finetune_path,
                      torch_dtype=torch.float16):
        """
        Load pipeline from fine-tuned weight

        """
        tokenizer = CLIPTokenizer.from_pretrained(
            os.path.join(finetune_path, "tokenizer"),
        )
        tokenizer_2 = CLIPTokenizer.from_pretrained(
            os.path.join(finetune_path, "tokenizer_2"),
        )

        # scheduler = DDIMScheduler(
        #     beta_start=0.00085,
        #     beta_end=0.012,
        #     beta_schedule="scaled_linear",
        #     clip_sample=False,
        #     set_alpha_to_one=False,
        # )

        pipeline: InteractEditPipeline = cls.from_pretrained(
            base_model,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            # scheduler=scheduler,  # we will use default scheduler
            torch_dtype=torch_dtype,
        )

        pipeline.load_textual_inversion(finetune_path)
        pipeline.load_lora_weights(os.path.join(finetune_path,"pytorch_lora_weights.safetensors"))

        pipeline.loaded = finetune_path

        return pipeline
        
    @staticmethod
    def detect_plot(image: Image.Image,
               sbj: str,
               obj: str,
               size: int,):
        
        image = image.copy()
        image.resize((size, size))
        
        gsam = GroundedSAM(segmenter="facebook/sam-vit-large")
        image_array, detections = gsam.grounded_segmentation(
            image,
            labels=[sbj, obj],
            polygon_refinement=True, # must be true
        )
        del gsam
        detected_image = annotate(image, detections)

        return detections, detected_image
        
    @classmethod
    def train(cls,
              base_model,
              output_dir,
              image: Image.Image,
              sbj: str,
              obj: str,
              action: str,
              initializer_tokens: List[str],
              detections: List[DetectionResult],
              initial_learning_rate=3e-4, # first stage
              learning_rate=1e-4,  # second stage
              text_encoder_lr=5e-6,
              train_batch_size=1,
              rank=64,
              text_encoder_rank=4,
              lambda_attention=0.01,
              phase1_train_steps=1000,
              phase2_train_steps=200,
              resolution=512,
              # optimizer
              adam_beta1=0.9,
              adam_beta2=0.999,
              adam_epsilon=1e-08,
              adam_weight_decay=1e-04,
              adam_weight_decay_text_encoder=1e-03,
              optimizer="AdamW", # or "prodigy"
              use_8bit_adam=False,
              # others
              allow_tf32=False,
              apply_masked_loss=True,
              center_crop=False,
              dataloader_num_workers=0,
              enable_xformers=False,
              gradient_accumulation_steps=1,
              lr_num_cycles=1,
              lr_power=1.0,
              lr_scheduler_type="constant",
              lr_warmup_steps=500,
              max_grad_norm=1.0,
              mixed_precision="fp16",
              num_of_assets=3,
              placeholder_token="<asset>",
              scale_lr=False,
              seed=None,
              use_dora=False,
              clip_skip: Optional[int] = None,
              lora_type: Literal["LoRA", "SeRA"] = "LoRA",
              gate_regularization_loss: bool = True,
              gate_reg_loss_temp: float = 1.0,
              gate_reg_loss_weight: float = 0.01,
              ):

        kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
            mixed_precision=mixed_precision,
            # project_config=accelerator_project_config,
            kwargs_handlers=[kwargs],
        )
        max_train_steps = phase1_train_steps + phase2_train_steps
        
        # Disable AMP for MPS.
        if torch.backends.mps.is_available():
            accelerator.native_amp = False
        
        # If passed along, set the training seed now.
        if seed is not None and seed != -1:
            set_seed(seed)
            
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            
        # load pretrained model, we assume it is already loaded using from_pretrained
        # here only prepare it for training
        # Load the tokenizers
        tokenizer_one = AutoTokenizer.from_pretrained(
            base_model,
            subfolder="tokenizer",
            use_fast=False,
        )
        tokenizer_two = AutoTokenizer.from_pretrained(
            base_model,
            subfolder="tokenizer_2",
            use_fast=False,
        )
        # import correct text encoder classes
        text_encoder_cls_one = import_model_class_from_model_name_or_path(
            base_model,
        )
        text_encoder_cls_two = import_model_class_from_model_name_or_path(
            base_model, subfolder="text_encoder_2"
        )
        text_encoder_one = text_encoder_cls_one.from_pretrained(
            base_model, subfolder="text_encoder",
        )
        text_encoder_two = text_encoder_cls_two.from_pretrained(
            base_model, subfolder="text_encoder_2",
        )
        noise_scheduler = DDPMScheduler.from_pretrained(base_model, subfolder="scheduler")
        
        vae = AutoencoderKL.from_pretrained(
            base_model,
            subfolder="vae",
        )
        latents_mean = latents_std = None
        if hasattr(vae.config, "latents_mean") and vae.config.latents_mean is not None:
            latents_mean = torch.tensor(vae.config.latents_mean).view(1, 4, 1, 1)
        if hasattr(vae.config, "latents_std") and vae.config.latents_std is not None:
            latents_std = torch.tensor(vae.config.latents_std).view(1, 4, 1, 1)
            
        unet = UNet2DConditionModel.from_pretrained(
            base_model, subfolder="unet",
        )
            
        placeholder_tokens = [
            placeholder_token.replace(">", f"{idx}>")
            for idx in range(num_of_assets)
        ]
        num_added_tokens_1 = tokenizer_one.add_tokens(placeholder_tokens)
        num_added_tokens_2 = tokenizer_two.add_tokens(placeholder_tokens)
        assert num_added_tokens_1 == num_added_tokens_2 == num_of_assets
        placeholder_token_ids_1 = tokenizer_one.convert_tokens_to_ids(
            placeholder_tokens
        )
        placeholder_token_ids_2 = tokenizer_two.convert_tokens_to_ids(
            placeholder_tokens
        )
        text_encoder_one.resize_token_embeddings(len(tokenizer_one))
        text_encoder_two.resize_token_embeddings(len(tokenizer_two))
        
        instance_prompt = "a photo of " + " and ".join(
            placeholder_tokens
        )
        
        if len(initializer_tokens) > 0:
            # Use initializer tokens
            token_embeds_one = text_encoder_one.get_input_embeddings().weight.data
            token_embeds_two = text_encoder_two.get_input_embeddings().weight.data
            for tkn_idx, initializer_token in enumerate(initializer_tokens):
                curr_token_ids_1 = tokenizer_one.encode(
                    initializer_token, add_special_tokens=False
                )
                curr_token_ids_2 = tokenizer_two.encode(
                    initializer_token, add_special_tokens=False
                )
                # assert (len(curr_token_ids)) == 1
                token_embeds_one[placeholder_token_ids_1[tkn_idx]] = token_embeds_one[
                    curr_token_ids_1[0]
                ]
                token_embeds_two[placeholder_token_ids_2[tkn_idx]] = token_embeds_two[
                    curr_token_ids_2[0]
                ]
        else:
            # Initialize new tokens randomly
            token_embeds_one = text_encoder_one.get_input_embeddings().weight.data
            token_embeds_two = text_encoder_two.get_input_embeddings().weight.data
            # token_embeds[-self.args.num_of_assets :] = token_embeds[
            #     -3 * self.args.num_of_assets : -2 * self.args.num_of_assets
            # ]
            token_embeds_one[-num_of_assets :] = token_embeds_one[
                -3 * num_of_assets : -2 * num_of_assets
            ]
            token_embeds_two[-num_of_assets :] = token_embeds_two[
                -3 * num_of_assets : -2 * num_of_assets
            ]
            
        # validation scheduler for logging
        validation_scheduler = DDIMScheduler(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                clip_sample=False,
                set_alpha_to_one=False,
            )
        validation_scheduler.set_timesteps(50)
        
        # Stage 1: We only train the additional adapter LoRA layers
        vae.requires_grad_(False)
        text_encoder_one.requires_grad_(False)
        text_encoder_two.requires_grad_(False)
        unet.requires_grad_(False)
        # train only text_input_embeds
        text_encoder_one.get_input_embeddings().requires_grad_(True)
        text_encoder_two.get_input_embeddings().requires_grad_(True)
        
        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
            # due to pytorch#99272, MPS does not yet support bfloat16.
            raise ValueError(
                "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
            )
        
        # Move unet, vae and text_encoder to device and cast to weight_dtype
        ## this line cause ValueError: Attempting to unscale FP16 gradients.
        unet.to(accelerator.device, dtype=weight_dtype)

        # The VAE is always in float32 to avoid NaN losses.
        vae.to(accelerator.device, dtype=torch.float32)

        ## this line cause ValueError: Attempting to unscale FP16 gradients.
        text_encoder_one.to(accelerator.device, dtype=weight_dtype)
        text_encoder_two.to(accelerator.device, dtype=weight_dtype)
        
        if enable_xformers:
            if is_xformers_available():
                import xformers

                xformers_version = version.parse(xformers.__version__)
                if xformers_version == version.parse("0.0.16"):
                    logger.warning(
                        "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, "
                        "please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                    )
                unet.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError("xformers is not available. Make sure it is installed correctly")
        
        def save_model(models, output_dir):
            if accelerator.is_main_process:
                # there are only two options here. Either are just the unet attn processor layers
                # or there are the unet and text encoder atten layers
                unet_lora_layers_to_save = None
                text_encoder_one_lora_layers_to_save = None
                text_encoder_two_lora_layers_to_save = None

                for model in models:
                    # here we set , save_embedding_layers=False because when it is set as auto,
                    # it will save token_embeddings of text_encoder, which cause error when
                    # loading the lora weights
                    if isinstance(model, type(accelerator.unwrap_model(unet))):
                        unet_lora_layers_to_save = convert_state_dict_to_diffusers(get_peft_model_state_dict(model))
                    elif isinstance(model, type(accelerator.unwrap_model(text_encoder_one))):
                        text_encoder_one_lora_layers_to_save = convert_state_dict_to_diffusers(
                            get_peft_model_state_dict(model, save_embedding_layers=False)
                        )
                    elif isinstance(model, type(accelerator.unwrap_model(text_encoder_two))):
                        text_encoder_two_lora_layers_to_save = convert_state_dict_to_diffusers(
                            get_peft_model_state_dict(model, save_embedding_layers=False)
                        )
                    else:
                        raise ValueError(f"unexpected save model: {model.__class__}")

                StableDiffusionXLPipeline.save_lora_weights(
                    output_dir,
                    unet_lora_layers=unet_lora_layers_to_save,
                    text_encoder_lora_layers=text_encoder_one_lora_layers_to_save,
                    text_encoder_2_lora_layers=text_encoder_two_lora_layers_to_save,
                )
                
                for model in models:
                    if isinstance(model, type(accelerator.unwrap_model(unet))):
                        pass
                    elif isinstance(model, type(accelerator.unwrap_model(text_encoder_one))):
                        weight_name = f"learned_embeds-steps-{global_step}.safetensors"
                        save_path = os.path.join(output_dir, weight_name)
                        save_progress(model, placeholder_token_ids_1, placeholder_tokens, accelerator, save_path, safe_serialization=True)
                    elif isinstance(model, type(accelerator.unwrap_model(text_encoder_two))):
                        weight_name = f"learned_embeds_2-steps-{global_step}.safetensors"
                        save_path = os.path.join(output_dir, weight_name)
                        save_progress(model, placeholder_token_ids_2, placeholder_tokens, accelerator, save_path, safe_serialization=True)
                    else:
                        raise ValueError(f"unexpected save model: {model.__class__}")
        
        # Enable TF32 for faster training on Ampere GPUs,
        # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
        if allow_tf32 and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            
        if scale_lr:
            learning_rate = (
                learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes
            )
            initial_learning_rate = (
                initial_learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes
            )
            
        # set params_to_optimize
        params_to_optimize = (
            itertools.chain(
                text_encoder_one.get_input_embeddings().parameters(),
                text_encoder_two.get_input_embeddings().parameters(),
            )
        )
        
        if mixed_precision == "fp16":
            models = [unet, text_encoder_one, text_encoder_two]
            # only upcast trainable parameters (LoRA) into fp32
            cast_training_params(models, dtype=torch.float32)
                        
        # Optimizer creation
        if not (optimizer.lower() == "adamw"):  # prodigy not implemented
            logger.warning(
                f"Unsupported choice of optimizer: {optimizer}.Supported optimizers include [adamW, prodigy]."
                "Defaulting to adamW"
            )
            optimizer = "adamw"
        
        if use_8bit_adam and not optimizer.lower() == "adamw":
            logger.warning(
                f"use_8bit_adam is ignored when optimizer is not set to 'AdamW'. Optimizer was "
                f"set to {optimizer.lower()}"
            )
            
        if optimizer.lower() == "adamw":
            if use_8bit_adam:
                try:
                    import bitsandbytes as bnb
                except ImportError:
                    raise ImportError(
                        "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                    )

                optimizer_class = bnb.optim.AdamW8bit
            else:
                optimizer_class = torch.optim.AdamW

            optimizer = optimizer_class(
                params_to_optimize,
                lr=initial_learning_rate,  # initial steps
                betas=(adam_beta1, adam_beta2),
                weight_decay=adam_weight_decay,
                eps=adam_epsilon,
            )
            
        def register_attention_control(controller):
            attn_procs = {}
            cross_att_count = 0
            for name in unet.attn_processors.keys():
                cross_attention_dim = (
                    None
                    if name.endswith("attn1.processor")
                    else unet.config.cross_attention_dim
                )
                if name.startswith("mid_block"):
                    hidden_size = unet.config.block_out_channels[-1]
                    place_in_unet = "mid"
                elif name.startswith("up_blocks"):
                    block_id = int(name[len("up_blocks.")])
                    hidden_size = list(reversed(unet.config.block_out_channels))[
                        block_id
                    ]
                    place_in_unet = "up"
                elif name.startswith("down_blocks"):
                    block_id = int(name[len("down_blocks.")])
                    hidden_size = unet.config.block_out_channels[block_id]
                    place_in_unet = "down"
                else:
                    continue
                cross_att_count += 1
                attn_procs[name] = P2PCrossAttnProcessor(
                    controller=controller, place_in_unet=place_in_unet
                )

            unet.set_attn_processor(attn_procs)
            controller.num_att_layers = cross_att_count
        
        # Dataset and DataLoaders creation:
        train_dataset = DreamBoothDataset(
            image=image,
            detections=detections,
            placeholder_tokens=placeholder_tokens,
            subject_object_label=[sbj, obj],
            action=action,
            size=(resolution,resolution),
            center_crop=center_crop,
            num_of_assets=num_of_assets,
        )
        
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            collate_fn=lambda examples: collate_fn(examples, resolution),
            num_workers=dataloader_num_workers,
        )
        
        def compute_time_ids(original_size, crops_coords_top_left):
            # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
            target_size = (resolution, resolution)
            add_time_ids = list(original_size + crops_coords_top_left + target_size)
            add_time_ids = torch.tensor([add_time_ids])
            add_time_ids = add_time_ids.to(accelerator.device, dtype=weight_dtype)
            return add_time_ids
        
        # if we're optimizing the text encoder (both if instance prompt is used
        # for all images or custom prompts) we need to tokenize and encode the
        # batch prompts on all training steps
        tokens_one = tokenize_prompt(tokenizer_one, instance_prompt)
        tokens_two = tokenize_prompt(tokenizer_two, instance_prompt)
        
         # Scheduler and math around the number of training steps.
        overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)

        lr_scheduler = get_scheduler(
            lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=lr_warmup_steps * accelerator.num_processes,
            num_training_steps=max_train_steps * accelerator.num_processes,
            num_cycles=lr_num_cycles,
            power=lr_power,
        )
        
        unet, text_encoder_one, text_encoder_two, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, text_encoder_one, text_encoder_two, optimizer, train_dataloader, lr_scheduler
        )
        num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)
        total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps
        
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_train_steps}")
        global_step = 0
        first_epoch = 0
        initial_global_step = 0
        
        progress_bar = tqdm(
            range(0, max_train_steps),
            initial=initial_global_step,
            desc="Steps",
            # Only show the progress bar once on each machine.
            disable=not accelerator.is_local_main_process,
        )
        
        # keep original embeddings as reference
        orig_embeds_params_one = (
            accelerator.unwrap_model(text_encoder_one)
            .get_input_embeddings()
            .weight.data.clone()
        )
        orig_embeds_params_two = (
            accelerator.unwrap_model(text_encoder_two)
            .get_input_embeddings()
            .weight.data.clone()
        )
        
        controller = AttentionStore()
        register_attention_control(controller)
        
        for epoch in range(first_epoch, num_train_epochs):
            unet.train()
            text_encoder_one.train()
            text_encoder_two.train()
            
            # set top parameter requires_grad = True for gradient checkpointing works
            accelerator.unwrap_model(text_encoder_one).text_model.embeddings.requires_grad_(True)
            accelerator.unwrap_model(text_encoder_two).text_model.embeddings.requires_grad_(True)
            
            for step, batch in enumerate(train_dataloader):
                # done stage 1
                if phase1_train_steps == global_step:
                    tokenizer_one.save_pretrained(
                        save_directory=os.path.join(output_dir, "tokenizer")
                    )
                    tokenizer_two.save_pretrained(
                        save_directory=os.path.join(output_dir, "tokenizer_2")
                    )

                    unet_lora_config = LoraConfig(
                        r=rank,
                        use_dora=use_dora,
                        lora_alpha=rank,
                        init_lora_weights="gaussian",
                        # target_modules=["attn2.to_k", "attn2.to_v"],
                        target_modules=["attn1.to_k", "attn1.to_v", "attn2.to_k", "attn2.to_v"],
                    )
                    if lora_type == "SeRA":
                        unet_lora_config._register_custom_module({nn.Linear: CustomLinear})
                    unet.add_adapter(unet_lora_config)
                    
                    text_lora_config = LoraConfig(
                        r=text_encoder_rank,
                        use_dora=use_dora,
                        lora_alpha=text_encoder_rank,
                        init_lora_weights="gaussian",
                        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
                    )
                    text_encoder_one.add_adapter(text_lora_config)
                    text_encoder_two.add_adapter(text_lora_config)
                    
                    if mixed_precision == "fp16":
                        models = [unet, text_encoder_one, text_encoder_two]
                        # only upcast trainable parameters (LoRA) into fp32
                        cast_training_params(models, dtype=torch.float32)
                    
                    unet_lora_parameters = list(filter(lambda p: p.requires_grad, unet.parameters()))
                    text_lora_parameters_one = list(filter(lambda p: p.requires_grad, text_encoder_one.parameters()))
                    text_lora_parameters_two = list(filter(lambda p: p.requires_grad, text_encoder_two.parameters()))
                    
                    if lora_type == "SeRA":
                        gating_params = []
                        for module in unet.modules():
                            if isinstance(module, CustomLinear):
                                gating_params += list(module.lora_G.values())
                        gating_ids = {id(p) for p in gating_params}
                        other_params = [p for p in unet_lora_parameters if id(p) not in gating_ids]

                    
                    # Optimization parameters
                    if lora_type == "SeRA":
                        unet_lora_parameters_with_lr    = [
                            {"params": other_params, "lr": learning_rate},
                            {"params": gating_params, "lr": learning_rate * 50, "weight_decay": 1e-4}
                        ]
                    elif lora_type == "LoRA":
                        unet_lora_parameters_with_lr    = [
                            {"params": unet_lora_parameters, "lr": learning_rate}
                        ]

                    text_lora_parameters_one_with_lr = {
                        "params": text_lora_parameters_one,
                        "weight_decay": adam_weight_decay_text_encoder,
                        "lr": text_encoder_lr if text_encoder_lr else learning_rate,
                    }
                    text_lora_parameters_two_with_lr = {
                        "params": text_lora_parameters_two,
                        "weight_decay": adam_weight_decay_text_encoder,
                        "lr": text_encoder_lr if text_encoder_lr else learning_rate,
                    }
                    text_input_embeds_with_lr = {
                        "params": itertools.chain(
                            text_encoder_one.get_input_embeddings().parameters(),
                            text_encoder_two.get_input_embeddings().parameters(),
                        ),
                        "weight_decay": adam_weight_decay_text_encoder,
                        "lr": text_encoder_lr if text_encoder_lr else learning_rate,
                    }
                    
                    params_to_optimize = [
                        text_lora_parameters_one_with_lr,
                        text_lora_parameters_two_with_lr,
                        text_input_embeds_with_lr,
                    ]
                    params_to_optimize.extend(unet_lora_parameters_with_lr)
                    
                    del optimizer
                    optimizer = optimizer_class(
                        params_to_optimize,
                        lr=learning_rate,
                        betas=(adam_beta1, adam_beta2),
                        weight_decay=adam_weight_decay,
                        eps=adam_epsilon,
                    )
                    del lr_scheduler
                    lr_scheduler = get_scheduler(
                        lr_scheduler_type,
                        optimizer=optimizer,
                        num_warmup_steps=lr_warmup_steps * accelerator.num_processes,
                        num_training_steps=max_train_steps * accelerator.num_processes,
                        num_cycles=lr_num_cycles,
                        power=lr_power,
                    )
                    optimizer, lr_scheduler, unet, text_encoder_one, text_encoder_two = accelerator.prepare(
                        optimizer, lr_scheduler, unet, text_encoder_one, text_encoder_two
                    )
                    # done prepare for stage 2
                
                logs = {}
                with accelerator.accumulate(unet):
                    pixel_values = batch["pixel_values"].to(dtype=vae.dtype)
                    prompts = batch["prompts"]
                    
                    # Convert images to latent space
                    model_input = vae.encode(pixel_values).latent_dist.sample()
                    
                    if latents_mean is None and latents_std is None:
                        model_input = model_input * vae.config.scaling_factor
                        # if pretrained_vae_model_name_or_path is None:
                        #     model_input = model_input.to(weight_dtype)
                    else:
                        latents_mean = latents_mean.to(device=model_input.device, dtype=model_input.dtype)
                        latents_std = latents_std.to(device=model_input.device, dtype=model_input.dtype)
                        model_input = (model_input - latents_mean) * vae.config.scaling_factor / latents_std
                        model_input = model_input.to(dtype=weight_dtype)
                    
                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(model_input)
                    bsz = model_input.shape[0]
                    
                    # Sample a random timestep for each image
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device
                    )
                    timesteps = timesteps.long()
                    
                    # Add noise to the model input according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)
                    
                    # time ids
                    add_time_ids = torch.cat(
                        [
                            compute_time_ids(original_size=s, crops_coords_top_left=c)
                            for s, c in zip(batch["original_sizes"], batch["crop_top_lefts"])
                        ]
                    )
                    
                    # Calculate the elements to repeat depending on the use of prior-preservation and custom captions.
                    elems_to_repeat_text_embeds = 1
                    
                    # Predict the noise residual
                    unet_added_conditions = {"time_ids": add_time_ids}
                    prompt_embeds, pooled_prompt_embeds, token_ids_tokenizer_1 = encode_prompt(
                        text_encoders=[text_encoder_one, text_encoder_two],
                        tokenizers=[tokenizer_one, tokenizer_two],
                        prompt=[instance_prompt], # or prompts[0]
                        # text_input_ids_list=[tokens_one, tokens_two],
                        clip_skip=clip_skip,
                    )
                    unet_added_conditions.update(
                        {"text_embeds": pooled_prompt_embeds.repeat(elems_to_repeat_text_embeds, 1)}
                    )
                    prompt_embeds_input = prompt_embeds.repeat(elems_to_repeat_text_embeds, 1, 1)
                    model_pred = unet(
                        noisy_model_input,
                        timesteps,
                        prompt_embeds_input,
                        added_cond_kwargs=unet_added_conditions,
                        return_dict=False,
                    )[0]
                    
                    # Get the target for loss depending on the prediction type
                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = (
                            noise_scheduler.get_velocity(model_input, noise, timesteps)
                        )
                    else:
                        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                     # compute masked loss
                     # TODO: need change this 64x64 to be dynamic
                    if apply_masked_loss:
                        if batch["instance_masks"].shape[1] == 0:
                            masks_shape= batch["instance_masks"].shape
                            downsampled_mask = torch.zeros([masks_shape[0], masks_shape[2], 64, 64], dtype=model_pred.dtype, device=model_pred.device)
                        else:
                            max_masks = torch.max(
                                batch["instance_masks"], axis=1
                            ).values
                            downsampled_mask = F.interpolate(
                                input=max_masks, size=(64, 64)
                            )
                        model_pred = model_pred * downsampled_mask
                        target = target * downsampled_mask
                    
                    # diffusion loss
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                    
                    if lora_type == "SeRA" and gate_regularization_loss and len(gating_params):
                        loss_gate = torch.tensor(0.0, device=loss.device, dtype=loss.dtype)
                        for p in gating_params:
                            gamma = torch.sigmoid(p * gate_reg_loss_temp)
                            loss_gate += (gamma * (1 - gamma)).mean()
                        if len(gating_params) > 0:  # avoid NaN
                            loss_gate = loss_gate / len(gating_params)
                        loss_gate = gate_reg_loss_weight * loss_gate
                        logs["gate_loss"] = loss_gate.detach().item()
                        loss = loss + loss_gate
                    
                    # Attention loss
                    if lambda_attention != 0:
                        attn_loss = torch.tensor(0., dtype=loss.dtype, device=loss.device)
                        for batch_idx in range(train_batch_size):
                            GT_masks = F.interpolate(
                                input=batch["instance_masks"][batch_idx], size=(16, 16)
                            )
                            agg_attn = aggregate_attention(
                                res=16,
                                from_where=("up", "down"),
                                is_cross=True,
                                select=batch_idx,
                                train_batch_size=train_batch_size,
                                controller=controller,
                            )
                            # no prior preservation batch
                            curr_cond_batch_idx = batch_idx
                            
                            for mask_id in range(len(GT_masks)):
                                curr_placeholder_token_id = placeholder_token_ids_1[
                                    batch["token_ids"][batch_idx][mask_id]
                                ]

                                asset_idx = (
                                    (
                                        tokens_one[curr_cond_batch_idx]  # input_ids/curr..
                                        == curr_placeholder_token_id
                                    )
                                    .nonzero()
                                    .item()
                                )
                                asset_attn_mask = agg_attn[..., asset_idx]
                                asset_attn_mask = (
                                    asset_attn_mask / asset_attn_mask.max()
                                )
                                attn_loss += F.mse_loss(
                                    GT_masks[mask_id, 0].float(),
                                    asset_attn_mask.float(),
                                    reduction="mean",
                                )

                        attn_loss = lambda_attention * (
                            attn_loss / train_batch_size
                        )
                        logs["attn_loss"] = attn_loss.detach().item()
                        loss += attn_loss
                    
                    accelerator.backward(loss)
                    
                    # No need to keep the attention store
                    controller.attention_store = {}
                    controller.cur_step = 0
                    
                    if accelerator.sync_gradients:
                        if phase1_train_steps > global_step:  # >
                            params_to_clip = (
                                itertools.chain(
                                    text_encoder_one.get_input_embeddings().parameters(),
                                    text_encoder_two.get_input_embeddings().parameters(),
                                )
                            )
                        else:
                            params_to_clip = (
                                itertools.chain(unet_lora_parameters, text_lora_parameters_one, text_lora_parameters_two)
                            )
                        accelerator.clip_grad_norm_(params_to_clip, max_grad_norm)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    
                    if global_step < phase1_train_steps:
                        # Let's make sure we don't update any embedding weights besides the newly added token
                        with torch.no_grad():
                            accelerator.unwrap_model(
                                text_encoder_one
                            ).get_input_embeddings().weight[
                                : -num_of_assets
                            ] = orig_embeds_params_one[
                                : -num_of_assets
                            ]
                            accelerator.unwrap_model(
                                text_encoder_two
                            ).get_input_embeddings().weight[
                                : -num_of_assets
                            ] = orig_embeds_params_two[
                                : -num_of_assets
                            ]
                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    
                logs["loss"] = loss.detach().item()
                logs["lr"] = lr_scheduler.get_last_lr()[0]
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
                
                if global_step >= max_train_steps:
                    break
            
        accelerator.wait_for_everyone()
        save_model(accelerator._models, output_dir)
             
        accelerator.end_training()
    
        return output_dir

# save_model(accelerator._models, ckpts_path)