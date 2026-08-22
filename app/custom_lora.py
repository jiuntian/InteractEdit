import gc
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import PIL
import numpy as np
from peft.tuners.lora.layer import LoraLayer, Linear
from peft.utils.integrations import gather_params_ctx
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)
from transformers.utils import (
    logging,
)
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, AutoencoderKL, StableDiffusionXLImg2ImgPipeline
from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders.lora_base import _fetch_state_dict, _load_lora_into_text_encoder, _func_optionally_disable_offloading
from diffusers.loaders.lora_pipeline import StableDiffusionXLLoraLoaderMixin, _LOW_CPU_MEM_USAGE_DEFAULT_LORA
from diffusers.loaders.peft import PeftAdapterMixin
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput
from diffusers.schedulers import KarrasDiffusionSchedulers, DDIMScheduler
from diffusers.utils import (
    convert_state_dict_to_diffusers,
    convert_state_dict_to_peft,
    logger, 
    is_invisible_watermark_available,
    is_transformers_version,
    scale_lora_layers,
    USE_PEFT_BACKEND,
    # check_peft_version,
    convert_unet_state_dict_to_peft,
    # delete_adapter_layers,
    get_adapter_name,
    # is_peft_available,
    is_peft_version,
    # set_adapter_layers,
    # set_weights_and_activate_adapters,
    # MIN_PEFT_VERSION,
)
from diffusers.utils.torch_utils import empty_device_cache, randn_tensor
from diffusers.utils.peft_utils import _create_lora_config, _maybe_warn_for_unhandled_keys

if is_invisible_watermark_available():
    from diffusers.pipelines.stable_diffusion_xl.watermark import StableDiffusionXLWatermarker


class CustomLoraLayer(LoraLayer):
    def __init__(self, base_layer, ephemeral_gpu_offload=False, **kwargs):
        super().__init__(base_layer, ephemeral_gpu_offload, **kwargs)
        logging.warning_once("Using custom LoRA Layer.")
        pass


class CustomLinear(Linear):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B", "lora_G") # need to add to avoid diff device when inference
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("r", "lora_alpha", "scaling", "lora_dropout")
    
    def __init__(self, base_layer, adapter_name, r=0, lora_alpha=1, lora_dropout=0, fan_in_fan_out=False, is_target_conv_1d_layer=False, init_lora_weights=True, use_rslora=False, use_dora=False, **kwargs):
        nn.Module.__init__(self)
        LoraLayer.__init__(self, base_layer, **kwargs)
        self.lora_G = nn.ParameterDict({})
        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            use_dora=use_dora,
        )
        self.is_target_conv_1d_layer = is_target_conv_1d_layer
        
    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                lora_G = self.lora_G[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = x.to(lora_A.weight.dtype)

                if not self.use_dora[active_adapter]:
                    # result = result + lora_B(lora_A(dropout(x))) * scaling
                    lora_G = torch.sigmoid(1 * lora_G)
                    result = result + lora_B(lora_A(dropout(x)) * lora_G) * scaling
                    
                else:
                    raise NotImplementedError("DoRA not supported yet!")

            result = result.to(torch_result_dtype)

        return result
    
    def update_layer(
        self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora, use_dora: bool = False
    ):
        # This code works for linear layers, override for other layer types
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        self.lora_A[adapter_name] = nn.Linear(self.in_features, r, bias=False)
        self.lora_B[adapter_name] = nn.Linear(r, self.out_features, bias=False)
        self.lora_G[adapter_name] = nn.Parameter(torch.zeros(r)) # use with sigmoid
        # self.lora_G[adapter_name] = nn.Parameter(torch.ones(r)) # use with sigmoid
        
        # change also in reset_lora_parameters()
        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = lora_alpha / r

        # for inits that require access to the base weight, use gather_param_ctx so that the weight is gathered when using DeepSpeed
        if isinstance(init_lora_weights, str) and init_lora_weights.startswith("pissa"):
            with gather_params_ctx(self.get_base_layer().weight):
                self.pissa_init(adapter_name, init_lora_weights)
        elif isinstance(init_lora_weights, str) and init_lora_weights.lower() == "olora":
            with gather_params_ctx(self.get_base_layer().weight):
                self.olora_init(adapter_name)
        elif init_lora_weights == "loftq":
            with gather_params_ctx(self.get_base_layer().weight):
                self.loftq_init(adapter_name)
        elif init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)
        # call this before dora_init
        self._move_adapter_to_device_of_base_layer(adapter_name)

        if use_dora:
            self.dora_init(adapter_name)
            self.use_dora[adapter_name] = True
        else:
            self.use_dora[adapter_name] = False

        self.set_adapter(self.active_adapters)
        
    def reset_lora_parameters(self, adapter_name, init_lora_weights):
        if init_lora_weights is False:
            return

        if adapter_name in self.lora_A.keys():
            # initialize the gate to 1
            # nn.init.ones_(self.lora_G[adapter_name]) # for raw
            nn.init.zeros_(self.lora_G[adapter_name]) # for sigmoid
            nn.init.kaiming_uniform_(self.lora_A[adapter_name].weight, a=math.sqrt(5))
            
            if init_lora_weights is True:
                # initialize A the same way as the default for nn.Linear and B to zero
                # https://github.com/microsoft/LoRA/blob/a0a92e0f26c067cf94747bdbf1ce73793fa44d19/loralib/layers.py#L124
                nn.init.kaiming_uniform_(self.lora_A[adapter_name].weight, a=math.sqrt(5))
            elif init_lora_weights.lower() == "gaussian":
                nn.init.normal_(self.lora_A[adapter_name].weight, std=1 / self.r[adapter_name])
            else:
                raise ValueError(f"Unknown initialization {init_lora_weights=}")
            nn.init.zeros_(self.lora_B[adapter_name].weight)
        if adapter_name in self.lora_embedding_A.keys():
            # Initialize A to zeros and B the same way as the default for nn.Embedding, see:
            # https://github.com/microsoft/LoRA/blob/4c0333854cb905966f8cc4e9a74068c1e507c7b7/loralib/layers.py#L59-L60
            nn.init.zeros_(self.lora_embedding_A[adapter_name])
            nn.init.normal_(self.lora_embedding_B[adapter_name])
            
    def _mixed_batch_forward(
        self, x: torch.Tensor, *args: Any, adapter_names: list[str], **kwargs: Any
    ) -> torch.Tensor:
        raise NotImplementedError("This works, but not up to date. Change before use")
    
    def get_delta_weight(self, adapter) -> torch.Tensor:
        raise NotImplementedError("Unable to merge/unmerge nor get_delta_weight")
    
    def unmerge(self) -> None:
        raise NotImplementedError("Unable to merge/unmerge nor get_delta_weight")
    
    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        raise NotImplementedError("Unable to merge/unmerge nor get_delta_weight")



## This being called by unet.load_lora_adapter
class CustomPeftAdapterMixin(PeftAdapterMixin):
    def load_lora_adapter(
        self, pretrained_model_name_or_path_or_dict, prefix="transformer", hotswap: bool = False,
        custom_lora:Dict = {}, **kwargs
    ):
        from peft import inject_adapter_in_model, set_peft_model_state_dict
        from peft.tuners.tuners_utils import BaseTunerLayer

        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", None)
        token = kwargs.pop("token", None)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", None)
        weight_name = kwargs.pop("weight_name", None)
        use_safetensors = kwargs.pop("use_safetensors", None)
        adapter_name = kwargs.pop("adapter_name", None)
        network_alphas = kwargs.pop("network_alphas", None)
        _pipeline = kwargs.pop("_pipeline", None)
        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", False)
        metadata = kwargs.pop("metadata", None)
        allow_pickle = False

        if low_cpu_mem_usage and is_peft_version("<=", "0.13.0"):
            raise ValueError(
                "`low_cpu_mem_usage=True` is not compatible with this `peft` version. Please update it with `pip install -U peft`."
            )

        user_agent = {"file_type": "attn_procs_weights", "framework": "pytorch"}
        state_dict, metadata = _fetch_state_dict(
            pretrained_model_name_or_path_or_dict=pretrained_model_name_or_path_or_dict,
            weight_name=weight_name,
            use_safetensors=use_safetensors,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            token=token,
            revision=revision,
            subfolder=subfolder,
            user_agent=user_agent,
            allow_pickle=allow_pickle,
            metadata=metadata,
        )
        if network_alphas is not None and prefix is None:
            raise ValueError("`network_alphas` cannot be None when `prefix` is None.")
        if network_alphas and metadata:
            raise ValueError("Both `network_alphas` and `metadata` cannot be specified.")

        if prefix is not None:
            state_dict = {k.removeprefix(f"{prefix}."): v for k, v in state_dict.items() if k.startswith(f"{prefix}.")}
            if metadata is not None:
                metadata = {k.removeprefix(f"{prefix}."): v for k, v in metadata.items() if k.startswith(f"{prefix}.")}

        if len(state_dict) > 0:
            if adapter_name in getattr(self, "peft_config", {}) and not hotswap:
                raise ValueError(
                    f"Adapter name {adapter_name} already in use in the model - please select a new adapter name."
                )
            elif adapter_name not in getattr(self, "peft_config", {}) and hotswap:
                raise ValueError(
                    f"Trying to hotswap LoRA adapter '{adapter_name}' but there is no existing adapter by that name. "
                    "Please choose an existing adapter name or set `hotswap=False` to prevent hotswapping."
                )

            # check with first key if is not in peft format
            first_key = next(iter(state_dict.keys()))
            if "lora_A" not in first_key:
                state_dict = convert_unet_state_dict_to_peft(state_dict)

            rank = {}
            for key, val in state_dict.items():
                # Cannot figure out rank from lora layers that don't have at least 2 dimensions.
                # Bias layers in LoRA only have a single dimension
                if "lora_B" in key and val.ndim > 1:
                    # Check out https://github.com/huggingface/peft/pull/2419 for the `^` symbol.
                    # We may run into some ambiguous configuration values when a model has module
                    # names, sharing a common prefix (`proj_out.weight` and `blocks.transformer.proj_out.weight`,
                    # for example) and they have different LoRA ranks.
                    rank[f"^{key}"] = val.shape[1]

            if network_alphas is not None and len(network_alphas) >= 1:
                alpha_keys = [k for k in network_alphas.keys() if k.startswith(f"{prefix}.")]
                network_alphas = {
                    k.removeprefix(f"{prefix}."): v for k, v in network_alphas.items() if k in alpha_keys
                }

            # create LoraConfig
            lora_config = _create_lora_config(state_dict, network_alphas, metadata, rank)
            
            if custom_lora:
                logger.info("INFO: register custom lora")
                lora_config._register_custom_module(custom_lora)

            # adapter_name
            if adapter_name is None:
                adapter_name = get_adapter_name(self)

            # <Unsafe code
            # We can be sure that the following works as it just sets attention processors, lora layers and puts all in the same dtype
            # Now we remove any existing hooks to `_pipeline`.

            # In case the pipeline has been already offloaded to CPU - temporarily remove the hooks
            # otherwise loading LoRA weights will lead to an error.
            is_model_cpu_offload, is_sequential_cpu_offload = self._optionally_disable_offloading(_pipeline)
            peft_kwargs = {}
            if is_peft_version(">=", "0.13.1"):
                peft_kwargs["low_cpu_mem_usage"] = low_cpu_mem_usage

            if hotswap or (self._prepare_lora_hotswap_kwargs is not None):
                if is_peft_version(">", "0.14.0"):
                    from peft.utils.hotswap import (
                        check_hotswap_configs_compatible,
                        hotswap_adapter_from_state_dict,
                        prepare_model_for_compiled_hotswap,
                    )
                else:
                    msg = (
                        "Hotswapping requires PEFT > v0.14. Please upgrade PEFT to a higher version or install it "
                        "from source."
                    )
                    raise ImportError(msg)

            if hotswap:

                def map_state_dict_for_hotswap(sd):
                    # For hotswapping, we need the adapter name to be present in the state dict keys
                    new_sd = {}
                    for k, v in sd.items():
                        if k.endswith("lora_A.weight") or key.endswith("lora_B.weight"):
                            k = k[: -len(".weight")] + f".{adapter_name}.weight"
                        elif k.endswith("lora_B.bias"):  # lora_bias=True option
                            k = k[: -len(".bias")] + f".{adapter_name}.bias"
                        new_sd[k] = v
                    return new_sd

            # To handle scenarios where we cannot successfully set state dict. If it's unsuccessful,
            # we should also delete the `peft_config` associated to the `adapter_name`.
            try:
                if hotswap:
                    state_dict = map_state_dict_for_hotswap(state_dict)
                    check_hotswap_configs_compatible(self.peft_config[adapter_name], lora_config)
                    try:
                        hotswap_adapter_from_state_dict(
                            model=self,
                            state_dict=state_dict,
                            adapter_name=adapter_name,
                            config=lora_config,
                        )
                    except Exception as e:
                        logger.error(f"Hotswapping {adapter_name} was unsuccessful with the following error: \n{e}")
                        raise
                    # the hotswap function raises if there are incompatible keys, so if we reach this point we can set
                    # it to None
                    incompatible_keys = None
                else:
                    inject_adapter_in_model(lora_config, self, adapter_name=adapter_name, **peft_kwargs)
                    incompatible_keys = set_peft_model_state_dict(self, state_dict, adapter_name, **peft_kwargs)

                    if self._prepare_lora_hotswap_kwargs is not None:
                        # For hotswapping of compiled models or adapters with different ranks.
                        # If the user called enable_lora_hotswap, we need to ensure it is called:
                        # - after the first adapter was loaded
                        # - before the model is compiled and the 2nd adapter is being hotswapped in
                        # Therefore, it needs to be called here
                        prepare_model_for_compiled_hotswap(
                            self, config=lora_config, **self._prepare_lora_hotswap_kwargs
                        )
                        # We only want to call prepare_model_for_compiled_hotswap once
                        self._prepare_lora_hotswap_kwargs = None

                # Set peft config loaded flag to True if module has been successfully injected and incompatible keys retrieved
                if not self._hf_peft_config_loaded:
                    self._hf_peft_config_loaded = True
            except Exception as e:
                # In case `inject_adapter_in_model()` was unsuccessful even before injecting the `peft_config`.
                if hasattr(self, "peft_config"):
                    for module in self.modules():
                        if isinstance(module, BaseTunerLayer):
                            active_adapters = module.active_adapters
                            for active_adapter in active_adapters:
                                if adapter_name in active_adapter:
                                    module.delete_adapter(adapter_name)

                    self.peft_config.pop(adapter_name)
                logger.error(f"Loading {adapter_name} was unsuccessful with the following error: \n{e}")
                raise

            _maybe_warn_for_unhandled_keys(incompatible_keys, adapter_name)

            # Offload back.
            if is_model_cpu_offload:
                _pipeline.enable_model_cpu_offload()
            elif is_sequential_cpu_offload:
                _pipeline.enable_sequential_cpu_offload()
            # Unsafe code />

        if prefix is not None and not state_dict:
            model_class_name = self.__class__.__name__
            logger.warning(
                f"No LoRA keys associated to {model_class_name} found with the {prefix=}. "
                "This is safe to ignore if LoRA state dict didn't originally have any "
                f"{model_class_name} related params. You can also try specifying `prefix=None` "
                "to resolve the warning. Otherwise, open an issue if you think it's unexpected: "
                "https://github.com/huggingface/diffusers/issues/new"
            )
            
## custom Unet
class CustomUNet2DConditionModel(UNet2DConditionModel, CustomPeftAdapterMixin):
    def test(self):
        return "test UNet2DConditionModel"     
    
           
## called by pipeline.load_lora_weights
class CustomStableDiffusionXLLoraLoaderMixin(StableDiffusionXLLoraLoaderMixin):
    def load_lora_weights(
        self,
        pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]],
        adapter_name: Optional[str] = None,
        hotswap: bool = False,
        custom_lora: Dict = {},
        custom_lora_text_encoder: Dict = {},
        **kwargs,
    ):
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for this method.")

        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", _LOW_CPU_MEM_USAGE_DEFAULT_LORA)
        if low_cpu_mem_usage and not is_peft_version(">=", "0.13.1"):
            raise ValueError(
                "`low_cpu_mem_usage=True` is not compatible with this `peft` version. Please update it with `pip install -U peft`."
            )

        # We could have accessed the unet config from `lora_state_dict()` too. We pass
        # it here explicitly to be able to tell that it's coming from an SDXL
        # pipeline.

        # if a dict is passed, copy it instead of modifying it inplace
        if isinstance(pretrained_model_name_or_path_or_dict, dict):
            pretrained_model_name_or_path_or_dict = pretrained_model_name_or_path_or_dict.copy()

        # First, ensure that the checkpoint is a compatible one and can be successfully loaded.
        kwargs["return_lora_metadata"] = True
        state_dict, network_alphas, metadata = self.lora_state_dict(
            pretrained_model_name_or_path_or_dict,
            unet_config=self.unet.config,
            **kwargs,
        )

        is_correct_format = all("lora" in key for key in state_dict.keys())
        if not is_correct_format:
            raise ValueError("Invalid LoRA checkpoint.")

        self.load_lora_into_unet(
            state_dict,
            network_alphas=network_alphas,
            unet=self.unet,
            adapter_name=adapter_name,
            metadata=metadata,
            _pipeline=self,
            low_cpu_mem_usage=low_cpu_mem_usage,
            hotswap=hotswap,
            custom_lora=custom_lora,
        )
        self.load_lora_into_text_encoder(
            state_dict,
            network_alphas=network_alphas,
            text_encoder=self.text_encoder,
            prefix=self.text_encoder_name,
            lora_scale=self.lora_scale,
            adapter_name=adapter_name,
            metadata=metadata,
            _pipeline=self,
            low_cpu_mem_usage=low_cpu_mem_usage,
            hotswap=hotswap,
            custom_lora=custom_lora_text_encoder,
        )
        self.load_lora_into_text_encoder(
            state_dict,
            network_alphas=network_alphas,
            text_encoder=self.text_encoder_2,
            prefix=f"{self.text_encoder_name}_2",
            lora_scale=self.lora_scale,
            adapter_name=adapter_name,
            metadata=metadata,
            _pipeline=self,
            low_cpu_mem_usage=low_cpu_mem_usage,
            hotswap=hotswap,
            custom_lora=custom_lora_text_encoder,
        )
        
    @classmethod
    # Copied from diffusers.loaders.lora_pipeline.StableDiffusionLoraLoaderMixin.load_lora_into_unet
    def load_lora_into_unet(
        cls,
        state_dict,
        network_alphas,
        unet: CustomUNet2DConditionModel,
        adapter_name=None,
        _pipeline=None,
        low_cpu_mem_usage=False,
        hotswap: bool = False,
        metadata=None,
        custom_lora: Dict = {},
    ):
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for this method.")

        if low_cpu_mem_usage and not is_peft_version(">=", "0.13.1"):
            raise ValueError(
                "`low_cpu_mem_usage=True` is not compatible with this `peft` version. Please update it with `pip install -U peft`."
            )

        # If the serialization format is new (introduced in https://github.com/huggingface/diffusers/pull/2918),
        # then the `state_dict` keys should have `cls.unet_name` and/or `cls.text_encoder_name` as
        # their prefixes.
        logger.info(f"Loading {cls.unet_name}.")
        unet.load_lora_adapter(
            state_dict,
            prefix=cls.unet_name,
            network_alphas=network_alphas,
            adapter_name=adapter_name,
            metadata=metadata,
            _pipeline=_pipeline,
            low_cpu_mem_usage=low_cpu_mem_usage,
            hotswap=hotswap,
            custom_lora=custom_lora,
        )

    @classmethod
    # Copied from diffusers.loaders.lora_pipeline.StableDiffusionLoraLoaderMixin.load_lora_into_text_encoder
    def load_lora_into_text_encoder(
        cls,
        state_dict,
        network_alphas,
        text_encoder,
        prefix=None,
        lora_scale=1.0,
        adapter_name=None,
        _pipeline=None,
        low_cpu_mem_usage=False,
        hotswap: bool = False,
        metadata=None,
        custom_lora: Dict = {},
    ):
        _load_custom_lora_into_text_encoder(
            state_dict=state_dict,
            network_alphas=network_alphas,
            lora_scale=lora_scale,
            text_encoder=text_encoder,
            prefix=prefix,
            text_encoder_name=cls.text_encoder_name,
            adapter_name=adapter_name,
            metadata=metadata,
            _pipeline=_pipeline,
            low_cpu_mem_usage=low_cpu_mem_usage,
            hotswap=hotswap,
            custom_lora=custom_lora,
        )
        
class CustomStableDiffusionXLPipeline(StableDiffusionXLPipeline, CustomStableDiffusionXLLoraLoaderMixin):
    def test(self):
        return "test CustomStableDiffusionXLPipeline"
    
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        text_encoder_2: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        tokenizer_2: CLIPTokenizer,
        unet: CustomUNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        image_encoder: CLIPVisionModelWithProjection = None,
        feature_extractor: CLIPImageProcessor = None,
        force_zeros_for_empty_prompt: bool = True,
        add_watermarker: Optional[bool] = None,
    ):
        # super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            unet=unet,
            scheduler=scheduler,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
        )
        self.register_to_config(force_zeros_for_empty_prompt=force_zeros_for_empty_prompt)
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

        self.default_sample_size = (
            self.unet.config.sample_size
            if hasattr(self, "unet") and self.unet is not None and hasattr(self.unet.config, "sample_size")
            else 128
        )

        add_watermarker = add_watermarker if add_watermarker is not None else is_invisible_watermark_available()

        if add_watermarker:
            self.watermark = StableDiffusionXLWatermarker()
        else:
            self.watermark = None 
            

class CustomStableDiffusionXLImg2ImgPipeline(StableDiffusionXLImg2ImgPipeline):
    @torch.inference_mode()
    def inversion(self,
                  prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        image: Union[
            torch.FloatTensor,
            PIL.Image.Image,
            np.ndarray,
            List[torch.FloatTensor],
            List[PIL.Image.Image],
            List[np.ndarray],
        ] = None,
        strength: float = 0.3,
        num_inference_steps: int = 50,
        denoising_start: Optional[float] = None,
        denoising_end: Optional[float] = None,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        original_size: Tuple[int, int] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Tuple[int, int] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
        aesthetic_score: float = 6.0,
        negative_aesthetic_score: float = 2.5,
    ):
        def _backward_ddim(x_tm1, alpha_t, alpha_tm1, eps_xt):
            """
            let a = alpha_t, b = alpha_{t - 1}
            We have a > b,
            x_{t} - x_{t - 1} = sqrt(a) ((sqrt(1/b) - sqrt(1/a)) * x_{t-1} + (sqrt(1/a - 1) - sqrt(1/b - 1)) * eps_{t-1})
            From https://arxiv.org/pdf/2105.05233.pdf, section F.
            """

            a, b = alpha_t, alpha_tm1
            sa = a**0.5
            sb = b**0.5

            return sa * ((1 / sb) * x_tm1 + ((1 / a - 1) ** 0.5 - (1 / b - 1) ** 0.5) * eps_xt)

        assert isinstance(self.scheduler, DDIMScheduler), 'scheduler must be DDIMScheduler to use inversion'

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            strength,
            num_inference_steps,
            callback_steps,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds,
            negative_prompt_embeds,
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = False
        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None)
            if cross_attention_kwargs is not None
            else None
        )
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )

        # 4. Preprocess image
        image = self.image_processor.preprocess(image)

        self.scheduler.set_timesteps(num_inference_steps, device=device)
       
        # 6. Prepare latent variables
        latents = self.prepare_latents(
            image,
            None,
            batch_size,
            num_images_per_prompt,
            prompt_embeds.dtype,
            device,
            generator,
            False,
        )
       
        height, width = latents.shape[-2:]
        height = height * self.vae_scale_factor
        width = width * self.vae_scale_factor

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # 8. Prepare added time ids & embeddings
        if negative_original_size is None:
            negative_original_size = original_size
        if negative_target_size is None:
            negative_target_size = target_size
            
        add_text_embeds = pooled_prompt_embeds
        if self.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.text_encoder_2.config.projection_dim
        add_time_ids, add_neg_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            aesthetic_score,
            negative_aesthetic_score,
            negative_original_size,
            negative_crops_coords_top_left,
            negative_target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        add_time_ids = add_time_ids.repeat(batch_size * num_images_per_prompt, 1)
        
        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device)
        
        added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
        prev_timestep = None

        for t in tqdm(reversed(self.scheduler.timesteps)):
            latent_model_input = latents
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                cross_attention_kwargs=cross_attention_kwargs,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            alpha_prod_t = self.scheduler.alphas_cumprod[t]
            alpha_prod_t_prev = (
                self.scheduler.alphas_cumprod[prev_timestep]
                if prev_timestep is not None
                else self.scheduler.final_alpha_cumprod
            )
            prev_timestep = t

            latents = _backward_ddim(
                x_tm1=latents,
                alpha_t=alpha_prod_t,
                alpha_tm1=alpha_prod_t_prev,
                eps_xt=noise_pred,
            )
            gc.collect()
            torch.cuda.empty_cache()
        image = latents
        return StableDiffusionXLPipelineOutput(images=image)
    
    
def _load_custom_lora_into_text_encoder(
    state_dict,
    network_alphas,
    text_encoder,
    prefix=None,
    lora_scale=1.0,
    text_encoder_name="text_encoder",
    adapter_name=None,
    _pipeline=None,
    low_cpu_mem_usage=False,
    hotswap: bool = False,
    metadata=None,
    custom_lora: Dict = {},
):
    if not USE_PEFT_BACKEND:
        raise ValueError("PEFT backend is required for this method.")

    if network_alphas and metadata:
        raise ValueError("`network_alphas` and `metadata` cannot be specified both at the same time.")

    peft_kwargs = {}
    if low_cpu_mem_usage:
        if not is_peft_version(">=", "0.13.1"):
            raise ValueError(
                "`low_cpu_mem_usage=True` is not compatible with this `peft` version. Please update it with `pip install -U peft`."
            )
        if not is_transformers_version(">", "4.45.2"):
            # Note from sayakpaul: It's not in `transformers` stable yet.
            # https://github.com/huggingface/transformers/pull/33725/
            raise ValueError(
                "`low_cpu_mem_usage=True` is not compatible with this `transformers` version. Please update it with `pip install -U transformers`."
            )
        peft_kwargs["low_cpu_mem_usage"] = low_cpu_mem_usage

    # If the serialization format is new (introduced in https://github.com/huggingface/diffusers/pull/2918),
    # then the `state_dict` keys should have `unet_name` and/or `text_encoder_name` as
    # their prefixes.
    prefix = text_encoder_name if prefix is None else prefix

    # Safe prefix to check with.
    if hotswap and any(text_encoder_name in key for key in state_dict.keys()):
        raise ValueError("At the moment, hotswapping is not supported for text encoders, please pass `hotswap=False`.")

    # Load the layers corresponding to text encoder and make necessary adjustments.
    if prefix is not None:
        state_dict = {k.removeprefix(f"{prefix}."): v for k, v in state_dict.items() if k.startswith(f"{prefix}.")}
        if metadata is not None:
            metadata = {k.removeprefix(f"{prefix}."): v for k, v in metadata.items() if k.startswith(f"{prefix}.")}

    if len(state_dict) > 0:
        logger.info(f"Loading {prefix}.")
        rank = {}
        state_dict = convert_state_dict_to_diffusers(state_dict)

        # convert state dict
        state_dict = convert_state_dict_to_peft(state_dict)

        for name, _ in text_encoder.named_modules():
            if name.endswith((".q_proj", ".k_proj", ".v_proj", ".out_proj", ".fc1", ".fc2")):
                rank_key = f"{name}.lora_B.weight"
                if rank_key in state_dict:
                    rank[rank_key] = state_dict[rank_key].shape[1]

        if network_alphas is not None:
            alpha_keys = [k for k in network_alphas.keys() if k.startswith(prefix) and k.split(".")[0] == prefix]
            network_alphas = {k.removeprefix(f"{prefix}."): v for k, v in network_alphas.items() if k in alpha_keys}

        # create `LoraConfig`
        lora_config = _create_lora_config(state_dict, network_alphas, metadata, rank, is_unet=False)

        if custom_lora:
            lora_config._register_custom_module(custom_lora)
            
        # adapter_name
        if adapter_name is None:
            adapter_name = get_adapter_name(text_encoder)

        # <Unsafe code
        is_model_cpu_offload, is_sequential_cpu_offload = _func_optionally_disable_offloading(_pipeline)
        # inject LoRA layers and load the state dict
        # in transformers we automatically check whether the adapter name is already in use or not
        text_encoder.load_adapter(
            adapter_name=adapter_name,
            adapter_state_dict=state_dict,
            peft_config=lora_config,
            **peft_kwargs,
        )

        # scale LoRA layers with `lora_scale`
        scale_lora_layers(text_encoder, weight=lora_scale)
        text_encoder.to(device=text_encoder.device, dtype=text_encoder.dtype)

        # Offload back.
        if is_model_cpu_offload:
            _pipeline.enable_model_cpu_offload()
        elif is_sequential_cpu_offload:
            _pipeline.enable_sequential_cpu_offload()
        # Unsafe code />

    if prefix is not None and not state_dict:
        model_class_name = text_encoder.__class__.__name__
        logger.warning(
            f"No LoRA keys associated to {model_class_name} found with the {prefix=}. "
            "This is safe to ignore if LoRA state dict didn't originally have any "
            f"{model_class_name} related params. You can also try specifying `prefix=None` "
            "to resolve the warning. Otherwise, open an issue if you think it's unexpected: "
            "https://github.com/huggingface/diffusers/issues/new"
        )
        