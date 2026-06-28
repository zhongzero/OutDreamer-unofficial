import decord
import os
import numpy as np
from torch import nn
import torch
from einops import rearrange, repeat
from typing import Any, Dict, Optional, Tuple
from torch.nn import functional as F
# from diffusers.models.transformer_2d import Transformer2DModelOutput
from diffusers.models.transformers.transformer_2d import Transformer2DModelOutput
from diffusers.utils import is_torch_version, deprecate
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormSingle
from diffusers.models.embeddings import PixArtAlphaTextProjection

from outdreamer.models.diffusion.outdreamer.modules import OverlapPatchEmbed3D, OverlapPatchEmbed2D, PatchEmbed2D, BasicTransformerBlock
from outdreamer.utils.utils import to_2tuple
try:
    import torch_npu
    from outdreamer.npu_config import npu_config
    from outdreamer.acceleration.parallel_states import get_sequence_parallel_state, hccl_info
except:
    torch_npu = None
    npu_config = None
    from outdreamer.utils.parallel_states import get_sequence_parallel_state, nccl_info

from PIL import Image
import numpy as np
from enum import Enum, auto
import glob

from outdreamer.models.diffusion.outdreamer.modeling_opensora import OpenSoraT2V

def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module

def reconstitute_checkpoint(pretrained_checkpoint, model_state_dict):
    pretrained_keys = set(list(pretrained_checkpoint.keys()))
    model_keys = set(list(model_state_dict.keys()))
    common_keys = list(pretrained_keys & model_keys)
    checkpoint = {k: pretrained_checkpoint[k] for k in common_keys if model_state_dict[k].numel() == pretrained_checkpoint[k].numel()}
    return checkpoint


class OpenSoraCNext(OpenSoraT2V):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        sample_size: Optional[int] = None,
        sample_size_t: Optional[int] = None,
        num_vector_embeds: Optional[int] = None,
        patch_size: Optional[int] = None,
        patch_size_t: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_type: str = "layer_norm",  # 'layer_norm', 'ada_norm', 'ada_norm_zero', 'ada_norm_single', 'ada_norm_continuous', 'layer_norm_i2vgen'
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        attention_type: str = "default",
        caption_channels: int = None,
        interpolation_scale_h: float = None,
        interpolation_scale_w: float = None,
        interpolation_scale_t: float = None,
        use_additional_conditions: Optional[bool] = None,
        attention_mode: str = 'xformers', 
        downsampler: str = None, 
        use_rope: bool = False,
        use_stable_fp32: bool = False,
        # inpaint
        # vae_scale_factor_t: int = 4,
        control_in_channels: Optional[int] = None,
    ):
        super().__init__(
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
            norm_num_groups=norm_num_groups,
            cross_attention_dim=cross_attention_dim,
            attention_bias=attention_bias,
            sample_size=sample_size,
            sample_size_t=sample_size_t,
            num_vector_embeds=num_vector_embeds,
            patch_size=patch_size,
            patch_size_t=patch_size_t,
            activation_fn=activation_fn,
            num_embeds_ada_norm=num_embeds_ada_norm,
            use_linear_projection=use_linear_projection,
            only_cross_attention=only_cross_attention,
            double_self_attention=double_self_attention,
            upcast_attention=upcast_attention,
            norm_type=norm_type,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
            attention_type=attention_type,
            caption_channels=caption_channels,
            interpolation_scale_h=interpolation_scale_h,
            interpolation_scale_w=interpolation_scale_w,
            interpolation_scale_t=interpolation_scale_t,
            use_additional_conditions=use_additional_conditions,
            attention_mode=attention_mode,
            downsampler=downsampler,
            use_rope=use_rope,
            use_stable_fp32=use_stable_fp32,
        )

        # self.vae_scale_factor_t = vae_scale_factor_t
        # init masked_video and mask conv_in
        # self._init_patched_inputs_for_inpainting()
        
        # light weight block for control condition
        # self.control_in_channels = in_channels + 1 if control_in_channels is None else control_in_channels
        self.control_in_channels = in_channels * 2 if control_in_channels is None else control_in_channels
        self.control_conv = nn.Sequential(
            nn.Conv2d(self.control_in_channels, 256, 3, stride=1, padding=1),
            nn.GroupNorm(in_channels, 256),
            nn.Conv2d(256, 512, 3, stride=2, padding=1), # downsample the (h, w) to (h//2, w//2), aligning to patch_size
            nn.GroupNorm(in_channels * 2, 512),
            nn.Conv2d(512, 1024, 3, stride=1, padding=1),
            nn.GroupNorm(in_channels * 4, 1024),
            nn.Conv2d(1024, cross_attention_dim, 3, stride=1, padding=1),
            nn.GroupNorm(in_channels * 8, cross_attention_dim),
        )
        # block0 [B, c, f, h, w]
    '''
    def _init_patched_inputs_for_inpainting(self):

        assert self.config.sample_size_t is not None, "OpenSoraCNext over patched input must provide sample_size_t"
        assert self.config.sample_size is not None, "OpenSoraCNext over patched input must provide sample_size"
        #assert not (self.config.sample_size_t == 1 and self.config.patch_size_t == 2), "Image do not need patchfy in t-dim"

        self.num_frames = self.config.sample_size_t
        self.config.sample_size = to_2tuple(self.config.sample_size)
        self.height = self.config.sample_size[0]
        self.width = self.config.sample_size[1]
        self.patch_size_t = self.config.patch_size_t
        self.patch_size = self.config.patch_size
        interpolation_scale_t = ((self.config.sample_size_t - 1) // 16 + 1) if self.config.sample_size_t % 2 == 1 else self.config.sample_size_t / 16
        interpolation_scale_t = (
            self.config.interpolation_scale_t if self.config.interpolation_scale_t is not None else interpolation_scale_t
        )
        interpolation_scale = (
            self.config.interpolation_scale_h if self.config.interpolation_scale_h is not None else self.config.sample_size[0] / 30, 
            self.config.interpolation_scale_w if self.config.interpolation_scale_w is not None else self.config.sample_size[1] / 40, 
        )
        
        if self.config.downsampler is not None and len(self.config.downsampler) == 9:
            self.pos_embed_mask = nn.ModuleList(
                [
                    OverlapPatchEmbed3D(
                        num_frames=self.config.sample_size_t,
                        height=self.config.sample_size[0],
                        width=self.config.sample_size[1],
                        patch_size_t=self.config.patch_size_t,
                        patch_size=self.config.patch_size,
                        in_channels=self.in_channels,
                        embed_dim=self.inner_dim,
                        interpolation_scale=interpolation_scale, 
                        interpolation_scale_t=interpolation_scale_t,
                        use_abs_pos=not self.config.use_rope, 
                    ),
                    zero_module(nn.Linear(self.inner_dim, self.inner_dim, bias=False)),
                ]
            )
            self.pos_embed_masked_video = nn.ModuleList(
                [
                    OverlapPatchEmbed3D(
                        num_frames=self.config.sample_size_t,
                        height=self.config.sample_size[0],
                        width=self.config.sample_size[1],
                        patch_size_t=self.config.patch_size_t,
                        patch_size=self.config.patch_size,
                        in_channels=self.in_channels,
                        embed_dim=self.inner_dim,
                        interpolation_scale=interpolation_scale, 
                        interpolation_scale_t=interpolation_scale_t,
                        use_abs_pos=not self.config.use_rope, 
                    ),
                    zero_module(nn.Linear(self.inner_dim, self.inner_dim, bias=False)),
                ]
            )
        elif self.config.downsampler is not None and len(self.config.downsampler) == 7:
            self.pos_embed_mask = nn.ModuleList(
                [
                    OverlapPatchEmbed2D(
                        num_frames=self.config.sample_size_t,
                        height=self.config.sample_size[0],
                        width=self.config.sample_size[1],
                        patch_size_t=self.config.patch_size_t,
                        patch_size=self.config.patch_size,
                        in_channels=self.in_channels,
                        embed_dim=self.inner_dim,
                        interpolation_scale=interpolation_scale, 
                        interpolation_scale_t=interpolation_scale_t,
                        use_abs_pos=not self.config.use_rope, 
                    ),
                    zero_module(nn.Linear(self.inner_dim, self.inner_dim, bias=False)),
                ]
            )
            
            self.pos_embed_masked_video = nn.ModuleList(
                [
                    OverlapPatchEmbed2D(
                        num_frames=self.config.sample_size_t,
                        height=self.config.sample_size[0],
                        width=self.config.sample_size[1],
                        patch_size_t=self.config.patch_size_t,
                        patch_size=self.config.patch_size,
                        in_channels=self.in_channels,
                        embed_dim=self.inner_dim,
                        interpolation_scale=interpolation_scale, 
                        interpolation_scale_t=interpolation_scale_t,
                        use_abs_pos=not self.config.use_rope, 
                    ),
                    zero_module(nn.Linear(self.inner_dim, self.inner_dim, bias=False)),
                ]
            )
        
        else:
            self.pos_embed_mask = nn.ModuleList(
                [
                    PatchEmbed2D(
                        num_frames=self.config.sample_size_t,
                        height=self.config.sample_size[0],
                        width=self.config.sample_size[1],
                        patch_size_t=self.config.patch_size_t,
                        patch_size=self.config.patch_size,
                        in_channels=self.in_channels,
                        embed_dim=self.inner_dim,
                        interpolation_scale=interpolation_scale, 
                        interpolation_scale_t=interpolation_scale_t,
                        use_abs_pos=not self.config.use_rope, 
                    ),
                    zero_module(nn.Linear(self.inner_dim, self.inner_dim, bias=False)),
                ]
            )
            self.pos_embed_masked_video = nn.ModuleList(
                [
                    PatchEmbed2D(
                        num_frames=self.config.sample_size_t,
                        height=self.config.sample_size[0],
                        width=self.config.sample_size[1],
                        patch_size_t=self.config.patch_size_t,
                        patch_size=self.config.patch_size,
                        in_channels=self.in_channels,
                        embed_dim=self.inner_dim,
                        interpolation_scale=interpolation_scale, 
                        interpolation_scale_t=interpolation_scale_t,
                        use_abs_pos=not self.config.use_rope, 
                    ),
                    zero_module(nn.Linear(self.inner_dim, self.inner_dim, bias=False)),
                ]
            )
    '''

    def _operate_on_patched_inputs(self, hidden_states, encoder_hidden_states, timestep, added_cond_kwargs, batch_size, frame, use_image_num):
        hidden_states_vid, hidden_states_img = self.pos_embed(hidden_states.to(self.dtype), frame)
        timestep_vid, timestep_img = None, None
        embedded_timestep_vid, embedded_timestep_img = None, None
        encoder_hidden_states_vid, encoder_hidden_states_img = None, None

        if self.adaln_single is not None:
            if self.use_additional_conditions and added_cond_kwargs is None:
                raise ValueError(
                    "`added_cond_kwargs` cannot be None when using additional conditions for `adaln_single`."
                )
            timestep, embedded_timestep = self.adaln_single(
                timestep, added_cond_kwargs, batch_size=batch_size, hidden_dtype=self.dtype
            )  # b 6d, b d
            if hidden_states_vid is None:
                timestep_img = timestep
                embedded_timestep_img = embedded_timestep
            else:
                timestep_vid = timestep
                embedded_timestep_vid = embedded_timestep
                if hidden_states_img is not None:
                    timestep_img = repeat(timestep, 'b d -> (b i) d', i=use_image_num).contiguous()
                    embedded_timestep_img = repeat(embedded_timestep, 'b d -> (b i) d', i=use_image_num).contiguous()

        if self.caption_projection is not None:
            encoder_hidden_states = self.caption_projection(encoder_hidden_states)  # b, 1+use_image_num, l, d or b, 1, l, d

            if hidden_states_vid is None:
                encoder_hidden_states_img = rearrange(encoder_hidden_states, 'b 1 l d -> (b 1) l d')
            else:
                encoder_hidden_states_vid = rearrange(encoder_hidden_states[:, :1], 'b 1 l d -> (b 1) l d')
                if hidden_states_img is not None:
                    encoder_hidden_states_img = rearrange(encoder_hidden_states[:, 1:], 'b i l d -> (b i) l d')


        return hidden_states_vid, hidden_states_img, encoder_hidden_states_vid, encoder_hidden_states_img, timestep_vid, timestep_img, embedded_timestep_vid, embedded_timestep_img
    
    def transformer_model_custom_load_state_dict(self, pretrained_model_path, load_mask=False):
        pretrained_model_path = os.path.join(pretrained_model_path, 'diffusion_pytorch_model.*')
        pretrained_model_path = glob.glob(pretrained_model_path)
        assert len(pretrained_model_path) > 0, f"Cannot find pretrained model in {pretrained_model_path}"
        pretrained_model_path = pretrained_model_path[0]

        print(f'Loading {self.__class__.__name__} pretrained weights...')
        print(f'Loading pretrained model from {pretrained_model_path}...')
        model_state_dict = self.state_dict()
        if 'safetensors' in pretrained_model_path:  # pixart series
            from safetensors.torch import load_file as safe_load
            # import ipdb;ipdb.set_trace()
            pretrained_checkpoint = safe_load(pretrained_model_path, device="cpu")
        else:  # latest stage training weight
            pretrained_checkpoint = torch.load(pretrained_model_path, map_location='cpu')
            if 'model' in pretrained_checkpoint:
                pretrained_checkpoint = pretrained_checkpoint['model']
        checkpoint = reconstitute_checkpoint(pretrained_checkpoint, model_state_dict)

        if not 'pos_embed_masked_video.0.weight' in checkpoint:
            checkpoint['pos_embed_masked_video.0.proj.weight'] = checkpoint['pos_embed.proj.weight']
            checkpoint['pos_embed_masked_video.0.proj.bias'] = checkpoint['pos_embed.proj.bias']
        if not 'pos_embed_mask.0.proj.weight' in checkpoint and load_mask:
            checkpoint['pos_embed_mask.0.proj.weight'] = checkpoint['pos_embed.proj.weight']
            checkpoint['pos_embed_mask.0.proj.bias'] = checkpoint['pos_embed.proj.bias']

        missing_keys, unexpected_keys = self.load_state_dict(checkpoint, strict=False)
        print(f'missing_keys {len(missing_keys)} {missing_keys}, unexpected_keys {len(unexpected_keys)}')
        print(f'Successfully load {len(self.state_dict()) - len(missing_keys)}/{len(model_state_dict)} keys from {pretrained_model_path}!')

    def custom_load_state_dict(self, pretrained_model_path, load_mask=False):
        assert isinstance(pretrained_model_path, dict), "pretrained_model_path must be a dict"

        pretrained_transformer_model_path = pretrained_model_path.get('transformer_model', None)

        self.transformer_model_custom_load_state_dict(pretrained_transformer_model_path, load_mask)

    def forward(
        self,
        hidden_states: torch.Tensor,
        control_cond: torch.Tensor,
        control_scale: float = 1.0,
        timestep: Optional[torch.LongTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        added_cond_kwargs: Dict[str, torch.Tensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        use_image_num: Optional[int] = 0,
        return_dict: bool = True,
    ):
        """
        The [`Transformer2DModel`] forward method.

        Args:
            hidden_states (`torch.LongTensor` of shape `(batch size, num latent pixels)` if discrete, `torch.FloatTensor` of shape `(batch size, channel, height, width)` if continuous):
                Input `hidden_states`.
            control_cond: (`torch.LongTensor` of shape `(batch size, control in_channels)` if discrete, `torch.FloatTensor` of shape `(batch size, channel, height, width)` if continuous):
                Control condition tensor.
            control_scale: (`float`):
                Scale factor for control condition tensor.
            encoder_hidden_states ( `torch.FloatTensor` of shape `(batch size, sequence len, embed dims)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            timestep ( `torch.LongTensor`, *optional*):
                Used to indicate denoising step. Optional timestep to be applied as an embedding in `AdaLayerNorm`.
            class_labels ( `torch.LongTensor` of shape `(batch size, num classes)`, *optional*):
                Used to indicate class labels conditioning. Optional class labels to be applied as an embedding in
                `AdaLayerZeroNorm`.
            cross_attention_kwargs ( `Dict[str, Any]`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            attention_mask ( `torch.Tensor`, *optional*):
                An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. If `1` the mask
                is kept, otherwise if `0` it is discarded. Mask will be converted into a bias, which adds large
                negative values to the attention scores corresponding to "discard" tokens.
            encoder_attention_mask ( `torch.Tensor`, *optional*):
                Cross-attention mask applied to `encoder_hidden_states`. Two formats supported:

                    * Mask `(batch, sequence_length)` True = keep, False = discard.
                    * Bias `(batch, 1, sequence_length)` 0 = keep, -10000 = discard.

                If `ndim == 2`: will be interpreted as a mask, then converted into a bias consistent with the format
                above. This bias will be added to the cross-attention scores.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unets.unet_2d_condition.UNet2DConditionOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        batch_size, c, frame, h, w = hidden_states.shape
        # print('hidden_states.shape', hidden_states.shape)
        frame = frame - use_image_num  # 21-4=17
        if cross_attention_kwargs is not None:
            if cross_attention_kwargs.get("scale", None) is not None:
                print.warning("Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")
        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension.
        #   we may have done this conversion already, e.g. if we came here via UNet2DConditionModel#forward.
        #   we can tell by counting dims; if ndim == 2: it's a mask rather than a bias.
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        attention_mask_vid, attention_mask_img = None, None
        if attention_mask is not None and attention_mask.ndim == 4:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #   (keep = +0,     discard = -10000.0)
            # b, frame+use_image_num, h, w -> a video with images
            # b, 1, h, w -> only images
            attention_mask = attention_mask.to(self.dtype)
            if get_sequence_parallel_state():
                if npu_config is not None:
                    attention_mask_vid = attention_mask[:, :frame * hccl_info.world_size]  # b, frame, h, w
                    attention_mask_img = attention_mask[:, frame * hccl_info.world_size:]  # b, use_image_num, h, w
                else:
                    # print('before attention_mask.shape', attention_mask.shape)
                    attention_mask_vid = attention_mask[:, :frame * nccl_info.world_size]  # b, frame, h, w
                    attention_mask_img = attention_mask[:, frame * nccl_info.world_size:]  # b, use_image_num, h, w
                    # print('after attention_mask.shape', attention_mask_vid.shape)
            else:
                attention_mask_vid = attention_mask[:, :frame]  # b, frame, h, w
                attention_mask_img = attention_mask[:, frame:]  # b, use_image_num, h, w

            if attention_mask_vid.numel() > 0:
                attention_mask_vid_first_frame = attention_mask_vid[:, :1].repeat(1, self.patch_size_t-1, 1, 1)
                attention_mask_vid = torch.cat([attention_mask_vid_first_frame, attention_mask_vid], dim=1)
                attention_mask_vid = attention_mask_vid.unsqueeze(1)  # b 1 t h w
                attention_mask_vid = F.max_pool3d(attention_mask_vid, kernel_size=(self.patch_size_t, self.patch_size, self.patch_size), 
                                                  stride=(self.patch_size_t, self.patch_size, self.patch_size))
                attention_mask_vid = rearrange(attention_mask_vid, 'b 1 t h w -> (b 1) 1 (t h w)') 
            if attention_mask_img.numel() > 0:
                attention_mask_img = F.max_pool2d(attention_mask_img, kernel_size=(self.patch_size, self.patch_size), stride=(self.patch_size, self.patch_size))
                attention_mask_img = rearrange(attention_mask_img, 'b i h w -> (b i) 1 (h w)') 

            attention_mask_vid = (1 - attention_mask_vid.bool().to(self.dtype)) * -10000.0 if attention_mask_vid.numel() > 0 else None
            attention_mask_img = (1 - attention_mask_img.bool().to(self.dtype)) * -10000.0 if attention_mask_img.numel() > 0 else None

            if frame == 1 and use_image_num == 0 and not get_sequence_parallel_state():
                attention_mask_img = attention_mask_vid
                attention_mask_vid = None
        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        # import ipdb;ipdb.set_trace()
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 3:  
            # b, 1+use_image_num, l -> a video with images
            # b, 1, l -> only images
            encoder_attention_mask = (1 - encoder_attention_mask.to(self.dtype)) * -10000.0
            in_t = encoder_attention_mask.shape[1]
            encoder_attention_mask_vid = encoder_attention_mask[:, :in_t-use_image_num]  # b, 1, l
            encoder_attention_mask_vid = rearrange(encoder_attention_mask_vid, 'b 1 l -> (b 1) 1 l') if encoder_attention_mask_vid.numel() > 0 else None

            encoder_attention_mask_img = encoder_attention_mask[:, in_t-use_image_num:]  # b, use_image_num, l
            encoder_attention_mask_img = rearrange(encoder_attention_mask_img, 'b i l -> (b i) 1 l') if encoder_attention_mask_img.numel() > 0 else None

            if frame == 1 and use_image_num == 0 and not get_sequence_parallel_state():
                encoder_attention_mask_img = encoder_attention_mask_vid
                encoder_attention_mask_vid = None

        if npu_config is not None and attention_mask_vid is not None:
            attention_mask_vid = npu_config.get_attention_mask(attention_mask_vid, attention_mask_vid.shape[-1])
            encoder_attention_mask_vid = npu_config.get_attention_mask(encoder_attention_mask_vid,
                                                                       attention_mask_vid.shape[-2])
        if npu_config is not None and attention_mask_img is not None:
            attention_mask_img = npu_config.get_attention_mask(attention_mask_img, attention_mask_img.shape[-1])
            encoder_attention_mask_img = npu_config.get_attention_mask(encoder_attention_mask_img,
                                                                       attention_mask_img.shape[-2])


        # 1. Input
        frame = ((frame - 1) // self.patch_size_t + 1) if frame % 2 == 1 else frame // self.patch_size_t  # patchfy
        # print('frame', frame)
        height, width = hidden_states.shape[-2] // self.patch_size, hidden_states.shape[-1] // self.patch_size

        added_cond_kwargs = {"resolution": None, "aspect_ratio": None}
        hidden_states_vid, hidden_states_img, encoder_hidden_states_vid, encoder_hidden_states_img, \
        timestep_vid, timestep_img, embedded_timestep_vid, embedded_timestep_img = self._operate_on_patched_inputs(
            hidden_states, encoder_hidden_states, timestep, added_cond_kwargs, batch_size, frame, use_image_num
        )

        # control cond
        control_cond = rearrange(control_cond, 'b c t h w -> (b t) c h w').contiguous()
        control_states = self.control_conv(control_cond)
        control_states = rearrange(control_states, '(b t) c h w -> (b t) (h w) c', b=batch_size).contiguous()
        control_states = rearrange(control_states, '(b t) n c -> b (t n) c', b=batch_size).contiguous()
        
        # 2. Blocks
        # import ipdb;ipdb.set_trace()
        if get_sequence_parallel_state():
            if hidden_states_vid is not None:
                # print(333333333333333)
                hidden_states_vid = rearrange(hidden_states_vid, 'b s h -> s b h', b=batch_size).contiguous()
                encoder_hidden_states_vid = rearrange(encoder_hidden_states_vid, 'b s h -> s b h',
                                                      b=batch_size).contiguous()
                timestep_vid = timestep_vid.view(batch_size, 6, -1).transpose(0, 1).contiguous()
                # print('timestep_vid', timestep_vid.shape)
                
                control_states = rearrange(control_states, 'b s h -> s b h', b=batch_size).contiguous()

        def align_mean_std(src, dst):
            '''
            e.g.: control_states = align_mean_std(control_states, dst=hidden_states_vid)
            '''
            mean_dst, std_dst = torch.mean(dst, dim=[1,2], keepdim=True), torch.std(dst, dim=[1,2], keepdim=True)
            mean_src, std_src = torch.mean(src, dim=[1,2], keepdim=True), torch.std(src, dim=[1,2], keepdim=True)
            src = (src - mean_src) / (std_src + 1e-9) * std_dst + mean_dst
            return src

        for index_block, block in enumerate(self.transformer_blocks):
        # for block in self.transformer_blocks:
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                # import ipdb;ipdb.set_trace()
                if hidden_states_vid is not None:
                    hidden_states_vid = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states_vid,
                        attention_mask_vid,
                        encoder_hidden_states_vid,
                        encoder_attention_mask_vid,
                        timestep_vid,
                        cross_attention_kwargs,
                        class_labels,
                        frame, 
                        height, 
                        width, 
                        **ckpt_kwargs,
                    )
                # import ipdb;ipdb.set_trace()
                if hidden_states_img is not None:
                    hidden_states_img = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states_img,
                        attention_mask_img,
                        encoder_hidden_states_img,
                        encoder_attention_mask_img,
                        timestep_img,
                        cross_attention_kwargs,
                        class_labels,
                        1, 
                        height, 
                        width, 
                        **ckpt_kwargs,
                    )
            else:
                if hidden_states_vid is not None:
                    hidden_states_vid = block(
                        hidden_states_vid,
                        attention_mask=attention_mask_vid,
                        encoder_hidden_states=encoder_hidden_states_vid,
                        encoder_attention_mask=encoder_attention_mask_vid,
                        timestep=timestep_vid,
                        cross_attention_kwargs=cross_attention_kwargs,
                        class_labels=class_labels,
                        frame=frame, 
                        height=height, 
                        width=width, 
                    )
                if hidden_states_img is not None:
                    hidden_states_img = block(
                        hidden_states_img,
                        attention_mask=attention_mask_img,
                        encoder_hidden_states=encoder_hidden_states_img,
                        encoder_attention_mask=encoder_attention_mask_img,
                        timestep=timestep_img,
                        cross_attention_kwargs=cross_attention_kwargs,
                        class_labels=class_labels,
                        frame=1, 
                        height=height, 
                        width=width, 
                    )

            if index_block == 0:
                if hidden_states_vid is not None:
                    control_states = align_mean_std(control_states, dst=hidden_states_vid)
                    hidden_states_vid = hidden_states_vid + control_states * control_scale # adjustable
                
                if hidden_states_img is not None:
                    control_states = align_mean_std(control_states, dst=hidden_states_img)
                    hidden_states_img = hidden_states_img + control_states * control_scale # adjustable


        if get_sequence_parallel_state():
            if hidden_states_vid is not None:
                hidden_states_vid = rearrange(hidden_states_vid, 's b h -> b s h', b=batch_size).contiguous()

        # 3. Output
        output_vid, output_img = None, None 
        if hidden_states_vid is not None:
            output_vid = self._get_output_for_patched_inputs(
                hidden_states=hidden_states_vid,
                timestep=timestep_vid,
                class_labels=class_labels,
                embedded_timestep=embedded_timestep_vid,
                num_frames=frame, 
                height=height,
                width=width,
            )  # b c t h w
        if hidden_states_img is not None:
            output_img = self._get_output_for_patched_inputs(
                hidden_states=hidden_states_img,
                timestep=timestep_img,
                class_labels=class_labels,
                embedded_timestep=embedded_timestep_img,
                num_frames=1, 
                height=height,
                width=width,
            )  # b c 1 h w
            if use_image_num != 0:
                output_img = rearrange(output_img, '(b i) c 1 h w -> b c i h w', i=use_image_num)

        if output_vid is not None and output_img is not None:
            output = torch.cat([output_vid, output_img], dim=2)
        elif output_vid is not None:
            output = output_vid
        elif output_img is not None:
            output = output_img

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

def OpenSoraCNext_S_122(**kwargs):
    return OpenSoraCNext(num_layers=28, attention_head_dim=96, num_attention_heads=16, patch_size_t=1, patch_size=2,
                       norm_type="ada_norm_single", caption_channels=4096, cross_attention_dim=1536, **kwargs)

def OpenSoraCNext_B_122(**kwargs):
    return OpenSoraCNext(num_layers=32, attention_head_dim=96, num_attention_heads=16, patch_size_t=1, patch_size=2,
                       norm_type="ada_norm_single", caption_channels=4096, cross_attention_dim=1920, **kwargs)

def OpenSoraCNext_L_122(**kwargs):
    return OpenSoraCNext(num_layers=40, attention_head_dim=128, num_attention_heads=16, patch_size_t=1, patch_size=2,
                       norm_type="ada_norm_single", caption_channels=4096, cross_attention_dim=2048, **kwargs)

def OpenSoraCNext_ROPE_L_122(**kwargs):
    return OpenSoraCNext(num_layers=32, attention_head_dim=96, num_attention_heads=24, patch_size_t=1, patch_size=2,
                       norm_type="ada_norm_single", caption_channels=4096, cross_attention_dim=2304, **kwargs)

OpenSoraCNext_models = {
    "OpenSoraCNext-S/122": OpenSoraCNext_S_122,  #       1.1B
    "OpenSoraCNext-B/122": OpenSoraCNext_B_122,
    "OpenSoraCNext-L/122": OpenSoraCNext_L_122,
    "OpenSoraCNext-ROPE-L/122": OpenSoraCNext_ROPE_L_122,
}

OpenSoraCNext_models_class = {
    "OpenSoraCNext-S/122": OpenSoraCNext,
    "OpenSoraCNext-B/122": OpenSoraCNext,
    "OpenSoraCNext-L/122": OpenSoraCNext,
    "OpenSoraCNext-ROPE-L/122": OpenSoraCNext,
}


if __name__ == '__main__':
    '''
    python -m outdreamer.models.diffusion.outdreamer.modeling_cnext
    '''
    # from outdreamer.models.causalvideovae import ae_channel_config, ae_stride_config
    # from outdreamer.models.causalvideovae import getae, getae_wrapper
    # from outdreamer.models.causalvideovae import CausalVAEModelWrapper

    args = type('args', (), 
    {
        'ae': 'CausalVAEModel_4x8x8', 
        'attention_mode': 'xformers', 
        'use_rope': True, 
        'model_max_length': 300, 
        'max_height': 320,
        'max_width': 240,
        'num_frames': 16,
        'use_image_num': 0, 
        'compress_kv_factor': 1, 
        'interpolation_scale_t': 1,
        'interpolation_scale_h': 1,
        'interpolation_scale_w': 1,
    }
    )
    b = 2
    c = 8
    cond_c = 4096
    num_timesteps = 1000
    ae_stride_t, ae_stride_h, ae_stride_w = 4, 8, 8 # ae_stride_config[args.ae]
    latent_size = (args.max_height // ae_stride_h, args.max_width // ae_stride_w)
    num_frames = (args.num_frames - 1) // ae_stride_t + 1

    device = torch.device('cuda:0')
    model = OpenSoraCNext_ROPE_L_122(in_channels=c, 
                              out_channels=c, 
                              sample_size=latent_size, 
                              sample_size_t=num_frames, 
                              activation_fn="gelu-approximate",
                            attention_bias=True,
                            attention_type="default",
                            double_self_attention=False,
                            norm_elementwise_affine=False,
                            norm_eps=1e-06,
                            norm_num_groups=32,
                            num_vector_embeds=None,
                            only_cross_attention=False,
                            upcast_attention=False,
                            use_linear_projection=False,
                            use_additional_conditions=False, 
                            downsampler=None,
                            interpolation_scale_t=args.interpolation_scale_t, 
                            interpolation_scale_h=args.interpolation_scale_h, 
                            interpolation_scale_w=args.interpolation_scale_w, 
                            use_rope=args.use_rope, 
                            ).to(device)
    
    # try:
    #     path = "PixArt-Alpha-XL-2-512.safetensors"
    #     from safetensors.torch import load_file as safe_load
    #     ckpt = safe_load(path, device="cpu")
    #     # import ipdb;ipdb.set_trace()
    #     if ckpt['pos_embed.proj.weight'].shape != model.pos_embed.proj.weight.shape and ckpt['pos_embed.proj.weight'].ndim == 4:
    #         repeat = model.pos_embed.proj.weight.shape[2]
    #         ckpt['pos_embed.proj.weight'] = ckpt['pos_embed.proj.weight'].unsqueeze(2).repeat(1, 1, repeat, 1, 1) / float(repeat)
    #         del ckpt['proj_out.weight'], ckpt['proj_out.bias']
    #     msg = model.load_state_dict(ckpt, strict=False)
    #     print(msg)
    # except Exception as e:
    #     print(e)
    print(model)
    print(f'{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e9} B')
    # import sys;sys.exit()
    x = torch.randn(b, c,  1+(args.num_frames-1)//ae_stride_t+args.use_image_num, args.max_height//ae_stride_h, args.max_width//ae_stride_w).to(device)
    cond = torch.randn(b, 1+args.use_image_num, args.model_max_length, cond_c).to(device)
    attn_mask = torch.randint(0, 2, (b, 1+(args.num_frames-1)//ae_stride_t+args.use_image_num, args.max_height//ae_stride_h, args.max_width//ae_stride_w)).to(device)  # B L or B 1+num_images L
    cond_mask = torch.randint(0, 2, (b, 1+args.use_image_num, args.model_max_length)).to(device)  # B L or B 1+num_images L
    timestep = torch.randint(0, 1000, (b,), device=device)
    b, c, f, h, w = x.shape
    control_cond = torch.ones((b, c + 1, f, h, w), dtype=x.dtype, device=x.device)
    control_scale = 1.0
    model_kwargs = dict(hidden_states=x, control_cond=control_cond, control_scale=control_scale, encoder_hidden_states=cond, attention_mask=attn_mask, 
                        encoder_attention_mask=cond_mask, use_image_num=args.use_image_num, timestep=timestep)
    with torch.no_grad():
        output = model(**model_kwargs)
    print(output[0].shape)