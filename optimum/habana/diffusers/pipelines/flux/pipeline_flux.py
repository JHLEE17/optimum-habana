# Copyright 2024 Black Forest Labs and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL.Image
import torch
import torch.nn as nn
# from para_attn.first_block_cache import utils
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention
from diffusers.models.autoencoders import AutoencoderKL
from diffusers.models.transformers import FluxTransformer2DModel
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline, calculate_shift, retrieve_timesteps
from diffusers.utils import BaseOutput, replace_example_docstring
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
from habana_frameworks.torch.hpu import wrap_in_hpu_graph

from optimum.utils import logging

from ....transformers.gaudi_configuration import GaudiConfig
from ....utils import HabanaProfile, speed_metrics, warmup_inference_steps_time_adjustment
from ...schedulers import GaudiFlowMatchEulerDiscreteScheduler
from ..pipeline_utils import GaudiDiffusionPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# Global flag to enable/disable FBCache analysis data collection
ENABLE_FB_ANALYSIS = False # Set to True to enable analysis
output_dir = "/workspace/jh/flux/outputs/analysis"


@dataclass
class GaudiFluxPipelineOutput(BaseOutput):
    """
    Output class for Stable Diffusion pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width,
            num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.
    """

    images: Union[List[PIL.Image.Image], np.ndarray]
    throughput: float


EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from optimum.habana.diffusers import GaudiFluxPipeline

        >>> pipe = GaudiFluxPipeline.from_pretrained(
        ...    "black-forest-labs/FLUX.1-schnell",
        ...     torch_dtype=torch.bfloat16,
        ...     use_habana=True,
        ...     use_hpu_graphs=True,
        ...     gaudi_config="Habana/stable-diffusion",
        ... )
        >>> prompt = "A cat holding a sign that says hello world"
        >>> # Depending on the variant being used, the pipeline call will slightly vary.
        >>> # Refer to the pipeline documentation for more details.
        >>> image = pipe(prompt, num_inference_steps=4, guidance_scale=0.0).images[0]
        >>> image.save("flux.png")
        ```
"""


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Adapted from: https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/models/embeddings.py#L697
    """
    cos_, sin_ = freqs_cis  # [S, D]

    cos = cos_[None, None]
    sin = sin_[None, None]
    cos, sin = cos.to(xq.device), sin.to(xq.device)

    xq_out = torch.ops.hpu.rotary_pos_embedding(xq, sin, cos, None, 0, 1)
    xk_out = torch.ops.hpu.rotary_pos_embedding(xk, sin, cos, None, 0, 1)

    return xq_out, xk_out


class GaudiFluxAttnProcessor2_0:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("FluxAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None:
            # `context` projections.
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)
            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)
            # attention
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)
        if image_rotary_emb is not None:
            query, key = apply_rotary_emb(query, key, image_rotary_emb)

        # hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        from habana_frameworks.torch.hpex.kernels import FusedSDPA

        hidden_states = FusedSDPA.apply(query, key, value, None, 0.0, False, None, "fast", None)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )
            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


class GaudiFusedFluxAttnProcessor2_0:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "FusedFluxAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        # `sample` projections.
        qkv = attn.to_qkv(hidden_states)
        split_size = qkv.shape[-1] // 3
        query, key, value = torch.split(qkv, split_size, dim=-1)
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        # `context` projections.
        if encoder_hidden_states is not None:
            encoder_qkv = attn.to_added_qkv(encoder_hidden_states)
            split_size = encoder_qkv.shape[-1] // 3
            (
                encoder_hidden_states_query_proj,
                encoder_hidden_states_key_proj,
                encoder_hidden_states_value_proj,
            ) = torch.split(encoder_qkv, split_size, dim=-1)
            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)
            # attention
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)
        if image_rotary_emb is not None:
            query, key = apply_rotary_emb(query, key, image_rotary_emb)

        # hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        from habana_frameworks.torch.hpex.kernels import FusedSDPA

        hidden_states = FusedSDPA.apply(query, key, value, None, 0.0, False, None, "fast", None)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )
            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


# 개선된 구현: first_block_graph + total_block_graph 접근 방식
# 1) 첫 번째 블록만 위한 래퍼 모듈 생성
class FirstBlockWrapper(nn.Module):
    def __init__(self, transformer):
        super().__init__()
        self.first_block = transformer.transformer_blocks[0]
        self.x_embedder = transformer.x_embedder
        self.time_text_embed = transformer.time_text_embed
        self.context_embedder = transformer.context_embedder
        self.pos_embed = transformer.pos_embed
    
    def forward(self, hidden_states, encoder_hidden_states, pooled_projections, timestep, 
                txt_ids, img_ids, guidance=None, joint_attention_kwargs=None):
        # 공통 임베딩 처리 (transformer 전처리 단계 포함)
        hidden_states_embedded = self.x_embedder(hidden_states)
        timestep = timestep.to(hidden_states_embedded.dtype) * 1000
        if guidance is None:
            temb = self.time_text_embed(timestep, pooled_projections)
        else:
            guidance = guidance.to(hidden_states_embedded.dtype) * 1000
            temb = self.time_text_embed(timestep, guidance, pooled_projections)
        encoder_hidden_states_embedded = self.context_embedder(encoder_hidden_states)
        
        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.pos_embed(ids)
        
        # 첫 번째 블록만 실행
        result = self.first_block(
            hidden_states_embedded, 
            encoder_hidden_states_embedded, 
            temb, 
            image_rotary_emb, 
            joint_attention_kwargs
        )
        
        # 결과, 원본 임베딩, 그리고 중간값들 반환
        return {
            'result': result,
            'hidden_states_embedded': hidden_states_embedded,
            'encoder_hidden_states_embedded': encoder_hidden_states_embedded,
            'temb': temb,
            'image_rotary_emb': image_rotary_emb
        }

class GaudiFluxPipeline(GaudiDiffusionPipeline, FluxPipeline):
    r"""
    Adapted from https://github.com/huggingface/diffusers/blob/v0.30.3/src/diffusers/pipelines/flux/pipeline_flux.py#L140
        Added batch size control for inference, and support for HPU graphs and Gaudi quantization via Intel Neural Compressor

    The Flux pipeline for text-to-image generation.

    Reference: https://blackforestlabs.ai/announcing-black-forest-labs/

    Args:
        transformer ([`FluxTransformer2DModel`]):
            Conditional Transformer (MMDiT) architecture to denoise the encoded image latents.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        text_encoder_2 ([`T5EncoderModel`]):
            [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically
            the [google/t5-v1_1-xxl](https://huggingface.co/google/t5-v1_1-xxl) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/en/model_doc/clip#transformers.CLIPTokenizer).
        tokenizer_2 (`T5TokenizerFast`):
            Second Tokenizer of class
            [T5TokenizerFast](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5TokenizerFast).
        use_habana (bool, defaults to `False`):
            Whether to use Gaudi (`True`) or CPU (`False`).
        use_hpu_graphs (bool, defaults to `False`):
            Whether to use HPU graphs or not.
        gaudi_config (Union[str, [`GaudiConfig`]], defaults to `None`):
            Gaudi configuration to use. Can be a string to download it from the Hub.
            Or a previously initialized config can be passed.
        bf16_full_eval (bool, defaults to `False`):
            Whether to use full bfloat16 evaluation instead of 32-bit.
            This will be faster and save memory compared to fp32/mixed precision but can harm generated images.
        sdp_on_bf16 (bool, defaults to `False`):
            Whether to allow PyTorch to use reduced precision in the SDPA math backend.
        rdt (float, defaults to -1.0):
            Residual difference threshold for FBCache analysis.
    """

    model_cpu_offload_seq = "text_encoder->text_encoder_2->transformer->vae"
    _optional_components = []
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(
        self,
        scheduler: GaudiFlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        text_encoder_2: T5EncoderModel,
        tokenizer_2: T5TokenizerFast,
        transformer: FluxTransformer2DModel,
        use_habana: bool = False,
        use_hpu_graphs: bool = False,
        gaudi_config: Union[str, GaudiConfig] = None,
        bf16_full_eval: bool = False,
        sdp_on_bf16: bool = False,
        rdt: float = -1.0,
    ):
        GaudiDiffusionPipeline.__init__(
            self,
            use_habana,
            use_hpu_graphs,
            gaudi_config,
            bf16_full_eval,
            sdp_on_bf16,
        )
        FluxPipeline.__init__(
            self,
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_encoder_2=text_encoder_2,
            tokenizer_2=tokenizer_2,
            transformer=transformer,
        )
        self.rdt = rdt # Store rdt for later checks

        for block in self.transformer.single_transformer_blocks:
            block.attn.processor = GaudiFluxAttnProcessor2_0()
        for block in self.transformer.transformer_blocks:
            block.attn.processor = GaudiFluxAttnProcessor2_0()

        self.to(self._device)
        self._has_been_quantized = False
        self.transformer_bf16 = None

        # Capture pipeline instance for closures
        pipeline_self = self

        if use_hpu_graphs and self.rdt > 0:
            # Initialize analysis data structure if analysis is enabled
            if ENABLE_FB_ANALYSIS:
                pipeline_self.fb_analysis_data = {
                    "timesteps": [],
                    "first_block_outputs": [],
                    "total_outputs": [],
                    "can_use_cache_flags": [],
                    "first_block_residuals": [],
                    "mean_diffs": [],
                    "mean_t1s": [],
                    "rdt": [],
                }

            # Import FirstBlockWrapper here if not already globally imported in this file
            # from .pipeline_flux import FirstBlockWrapper # Assuming it's defined in the same file or accessible

            # FirstBlockWrapper should be created from the original transformer module
            pipeline_self.first_block_wrapper_for_cache = FirstBlockWrapper(transformer)
            pipeline_self.first_block_hpu_graph_for_cache = wrap_in_hpu_graph(pipeline_self.first_block_wrapper_for_cache)
            
            # Wrap the main transformer for HPU graph execution
            main_transformer_hpu_graph = wrap_in_hpu_graph(transformer)
            # Capture the original forward method of the HPU-wrapped main transformer
            # This is the method that will be called if cache is not used for the whole model.
            original_main_transformer_hpu_graph_forward = main_transformer_hpu_graph.forward
            
            # 3) 전체 모델 그래프 생성 (원래 transformer를 그대로 사용)
            # 원본 forward 함수를 저장 (이미 original_main_transformer_hpu_graph_forward 로 저장됨)

            # 4) FirstBlockCache + HPU 그래프 통합 forward 함수
            def graph_cached_forward(transformer_instance_self, hidden_states, encoder_hidden_states=None, pooled_projections=None, timestep=None, img_ids=None, txt_ids=None, guidance=None, joint_attention_kwargs=None, return_dict=False, **kwargs):
                # transformer_instance_self is the main_transformer_hpu_graph instance
                import habana_frameworks.torch.core as htcore
                from para_attn.first_block_cache.utils import get_can_use_cache, set_buffer, get_buffer, apply_prev_hidden_states_residual, mark_step_begin
                
                # 원본 입력 형태 저장
                original_shape = hidden_states.shape
                
                # 1) 캐싱 단계 표시 및 스텝 시작
                mark_step_begin()
                
                if ENABLE_FB_ANALYSIS:
                    # 현재 timestep 저장 (분석용)
                    current_timestep_val = timestep[0].item() if timestep.ndim > 0 else timestep.item()
                    pipeline_self.fb_analysis_data["timesteps"].append(current_timestep_val)

                # 2) first_block_graph 실행 (임베딩 + 첫 번째 블록)
                # 파이프라인 레벨의 first_block_graph 사용 (captured pipeline_self)
                first_block_outputs_dict = pipeline_self.first_block_hpu_graph_for_cache(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    pooled_projections=pooled_projections,
                    timestep=timestep,
                    txt_ids=txt_ids,
                    img_ids=img_ids,
                    guidance=guidance,
                    joint_attention_kwargs=joint_attention_kwargs
                )
                htcore.mark_step()
                
                # 결과 및 중간값 추출
                first_block_result = first_block_outputs_dict['result'] 
                hidden_states_embedded = first_block_outputs_dict['hidden_states_embedded']
                # encoder_hidden_states_embedded = first_block_outputs_dict['encoder_hidden_states_embedded']
                
                if not isinstance(first_block_result, torch.Tensor):
                    # GaudiFluxTransformerBlock returns (encoder_hidden_states, hidden_states)
                    # We are interested in the hidden_states part from the first block.
                    _, hidden_states_first = first_block_result
                else:
                    # This case might not be hit if FirstBlockWrapper always wraps a FluxTransformerBlock
                    hidden_states_first = first_block_result
                
                if ENABLE_FB_ANALYSIS:
                    # 첫 번째 블록 출력 저장 (분석용)
                    pipeline_self.fb_analysis_data["first_block_outputs"].append(hidden_states_first.cpu().clone().detach())
                
                # 3) 첫 번째 블록 residual 계산
                first_hidden_states_residual = hidden_states_first - hidden_states_embedded
                if ENABLE_FB_ANALYSIS:
                    pipeline_self.fb_analysis_data["first_block_residuals"].append(first_hidden_states_residual.cpu().clone().detach())
                
                # 4) 캐시 사용 가능 여부 확인
                # The underlying transformer module for _is_parallelized might be nested if HPU graph is applied
                base_transformer_module = transformer_instance_self.module if hasattr(transformer_instance_self, 'module') else transformer_instance_self
                can_use_cache = get_can_use_cache(
                    first_hidden_states_residual,
                    parallelized=getattr(base_transformer_module, "_is_parallelized", False),
                )[0]
                # can_use_cache = can_use_cache_diff[0]
                # mean_diff = can_use_cache_diff[1]
                # mean_t1 = can_use_cache_diff[2]
                if timestep[0].item() > 0.82 and timestep[0].item() < 0.84:
                    can_use_cache = False
                if ENABLE_FB_ANALYSIS:
                    pipeline_self.fb_analysis_data["can_use_cache_flags"].append(can_use_cache)
                    # pipeline_self.fb_analysis_data["mean_diffs"].append(mean_diff)
                    # pipeline_self.fb_analysis_data["mean_t1s"].append(mean_t1)
                    # pipeline_self.fb_analysis_data["rdt"].append([self.rdt, mean_diff/mean_t1 if mean_t1 is not None else None])
                htcore.mark_step()
                # print(f"can_use_cache: {can_use_cache}")
                
                # 5) 캐싱 로직 적용
                if can_use_cache:
                    # 캐시된 결과 사용
                    del first_hidden_states_residual
                    # 이전에 저장된 전체 결과에 residual 적용
                    total_output = get_buffer("total_output")
                    assert total_output is not None, "total_output must be set before using cache"
                else:
                    # 첫 번째 블록 residual 저장
                    set_buffer("first_hidden_states_residual", first_hidden_states_residual)
                    del first_hidden_states_residual
                    
                    # 6) 전체 transformer 그래프 실행
                    # 캡처된 original_main_transformer_hpu_graph_forward 호출
                    # This was the .forward of main_transformer_hpu_graph before it was replaced.
                    # It's already bound to main_transformer_hpu_graph if it was a method.
                    # The arguments passed to graph_cached_forward (excluding transformer_instance_self)
                    # are the ones that original_main_transformer_hpu_graph_forward expects.
                    total_output = original_main_transformer_hpu_graph_forward(
                        hidden_states, # Start with actual arguments
                        encoder_hidden_states,
                        pooled_projections,
                        timestep,
                        img_ids,
                        txt_ids,
                        guidance,
                        joint_attention_kwargs,
                        return_dict=False, # Match original call signature if it differs
                    )[0]
                
                # 결과 저장
                if ENABLE_FB_ANALYSIS:
                    pipeline_self.fb_analysis_data["total_outputs"].append(total_output.cpu().clone().detach())
                set_buffer("total_output", total_output)
                
                htcore.mark_step()
                
                # 7) 결과 확인 및 반환
                assert total_output.shape == original_shape, f"Output shape mismatch: {total_output.shape} != {original_shape}"
                
                if not return_dict:
                    return (total_output,)
                
                from diffusers.models.transformers.transformer_flux import Transformer2DModelOutput
                return Transformer2DModelOutput(sample=total_output)

            # 새 forward 함수 연결 (main_transformer_hpu_graph의 forward를 교체)
            main_transformer_hpu_graph.forward = graph_cached_forward.__get__(main_transformer_hpu_graph, type(main_transformer_hpu_graph))
            self.transformer = main_transformer_hpu_graph # Assign the HPU graph wrapped transformer to self.transformer
            
            # For FBCache analysis
            self.fb_analysis_data = {
                "timesteps": [],
                "first_block_outputs": [],
                "total_outputs": [],
                "can_use_cache_flags": [],
                "first_block_residuals": [], # FBCache 판단 기준
                "mean_diffs": [],
                "mean_t1s": [],
                "rdt": [],
            }
        elif use_hpu_graphs:
            # 'transformer' here is the original nn.Module passed to __init__
            self.transformer = wrap_in_hpu_graph(transformer)
        else:
            # No HPU graphs, self.transformer is the original nn.Module
            self.transformer = transformer
            pass

    @classmethod
    def _split_inputs_into_batches(cls, batch_size, latents, prompt_embeds, pooled_prompt_embeds, guidance):
        # Use torch.split to generate num_batches batches of size batch_size
        latents_batches = list(torch.split(latents, batch_size))
        prompt_embeds_batches = list(torch.split(prompt_embeds, batch_size))
        if pooled_prompt_embeds is not None:
            pooled_prompt_embeds_batches = list(torch.split(pooled_prompt_embeds, batch_size))
        if guidance is not None:
            guidance_batches = list(torch.split(guidance, batch_size))

        # If the last batch has less samples than batch_size, pad it with dummy samples
        num_dummy_samples = 0
        if latents_batches[-1].shape[0] < batch_size:
            num_dummy_samples = batch_size - latents_batches[-1].shape[0]

            # Pad latents_batches
            sequence_to_stack = (latents_batches[-1],) + tuple(
                torch.zeros_like(latents_batches[-1][0][None, :]) for _ in range(num_dummy_samples)
            )
            latents_batches[-1] = torch.vstack(sequence_to_stack)

            # Pad prompt_embeds_batches
            sequence_to_stack = (prompt_embeds_batches[-1],) + tuple(
                torch.zeros_like(prompt_embeds_batches[-1][0][None, :]) for _ in range(num_dummy_samples)
            )
            prompt_embeds_batches[-1] = torch.vstack(sequence_to_stack)

            # Pad pooled_prompt_embeds if necessary
            if pooled_prompt_embeds is not None:
                sequence_to_stack = (pooled_prompt_embeds_batches[-1],) + tuple(
                    torch.zeros_like(pooled_prompt_embeds_batches[-1][0][None, :]) for _ in range(num_dummy_samples)
                )
                pooled_prompt_embeds_batches[-1] = torch.vstack(sequence_to_stack)

            # Pad guidance if necessary
            if guidance is not None:
                guidance_batches[-1] = guidance_batches[-1].unsqueeze(1)
                sequence_to_stack = (guidance_batches[-1],) + tuple(
                    torch.zeros_like(guidance_batches[-1][0][None, :]) for _ in range(num_dummy_samples)
                )
                guidance_batches[-1] = torch.vstack(sequence_to_stack).squeeze(1)

        # Stack batches in the same tensor
        latents_batches = torch.stack(latents_batches)
        prompt_embeds_batches = torch.stack(prompt_embeds_batches)
        pooled_prompt_embeds_batches = torch.stack(pooled_prompt_embeds_batches)
        guidance_batches = torch.stack(guidance_batches) if guidance is not None else None

        return (
            latents_batches,
            prompt_embeds_batches,
            pooled_prompt_embeds_batches,
            guidance_batches,
            num_dummy_samples,
        )

    def quantize(self, quant_mode, quant_config_path=None):
        """
        Quantize the pipeline using neural compressor.

        Args:
            quant_mode (`str`):
                The quantization mode. Can be 'measure', 'quantize', or 'quantize-mixed'.
            quant_config_path (`str`, *optional*):
                Path to the quantization configuration JSON file. If not provided, it will try to get from QUANT_CONFIG env variable.
        
        Returns:
            self: The pipeline itself for chaining.
        """
        # measure 모드는 __call__에서 처리
        if quant_mode == "measure":
            return self
            
        import habana_frameworks.torch.core as htcore

        if quant_mode == "quantize-mixed":
            import copy
            self.transformer_bf16 = copy.deepcopy(self.transformer).to(self._execution_device)

        if quant_mode in ("quantize", "quantize-mixed"):
            import os

            config_path = quant_config_path
            if not config_path:
                config_path = os.getenv("QUANT_CONFIG")
            
            if not config_path:
                raise ImportError(
                    "QUANT_CONFIG path is not defined. Please provide quant_config_path or define QUANT_CONFIG environment variable."
                )
            elif not os.path.isfile(config_path):
                raise ImportError(f"Quantization config path '{config_path}' is not valid")

            htcore.hpu_set_env()

            from neural_compressor.torch.quantization import FP8Config, convert, prepare
            from .pipeline_flux import FirstBlockWrapper
            
            config = FP8Config.from_json_file(config_path)

            # Determine the base nn.Module for quantization and for FirstBlockWrapper updates
            base_transformer_module_for_quant = self.transformer.module if hasattr(self.transformer, 'module') and isinstance(self.transformer.module, torch.nn.Module) else self.transformer
            
            if quant_mode == "quantize":
                self.transformer = convert(base_transformer_module_for_quant, config)
                current_underlying_module = self.transformer.module if hasattr(self.transformer, 'module') and isinstance(self.transformer.module, torch.nn.Module) else self.transformer

                if hasattr(self, "first_block_wrapper_for_cache"):
                    self.first_block_wrapper_for_cache = FirstBlockWrapper(current_underlying_module)
                    self.first_block_hpu_graph_for_cache = wrap_in_hpu_graph(self.first_block_wrapper_for_cache)
            
            # Initialize the main transformer
            htcore.hpu_initialize(self.transformer, mark_only_scales_as_const=True)
        
        self._has_been_quantized = True        
        return self

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        timesteps: List[int] = None,
        guidance_scale: float = 3.5,
        batch_size: int = 1,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        profiling_warmup_steps: Optional[int] = 0,
        profiling_steps: Optional[int] = 0,
        **kwargs,
    ):
        r"""
        Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux/pipeline_flux.py#L531
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                will be used instead
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 7.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.flux.FluxPipelineOutput`] instead of a plain tuple.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int` defaults to 512): Maximum sequence length to use with the `prompt`.
            profiling_warmup_steps (`int`, *optional*):
                Number of steps to ignore for profling.
            profiling_steps (`int`, *optional*):
                Number of steps to be captured when enabling profiling.

        Examples:

        Returns:
            [`~pipelines.flux.FluxPipelineOutput`] or `tuple`: [`~pipelines.flux.FluxPipelineOutput`] if `return_dict`
            is True, otherwise a `tuple`. When returning a tuple, the first element is a list with the generated
            images.
        """
        import habana_frameworks.torch as ht
        import habana_frameworks.torch.core as htcore

        quant_mode = kwargs.get("quant_mode", None)
        
        # measure 모드는 기존 방식대로 __call__에서 처리
        if quant_mode == "measure":
            import os

            quant_config_path = os.getenv("QUANT_CONFIG")
            if not quant_config_path:
                raise ImportError(
                    "QUANT_CONFIG path is not defined. Please define path to quantization configuration JSON file."
                )
            elif not os.path.isfile(quant_config_path):
                raise ImportError(f"QUANT_CONFIG path '{quant_config_path}' is not valid")

            htcore.hpu_set_env()

            from neural_compressor.torch.quantization import FP8Config, convert, prepare
            from .pipeline_flux import FirstBlockWrapper
            
            config = FP8Config.from_json_file(quant_config_path)

            # Determine the base nn.Module for quantization and for FirstBlockWrapper updates
            base_transformer_module_for_quant = self.transformer.module if hasattr(self.transformer, 'module') and isinstance(self.transformer.module, torch.nn.Module) else self.transformer
            
            # prepare can modify in-place or return a new object
            self.transformer = prepare(base_transformer_module_for_quant, config)
            
            # Get the actual prepared module for FirstBlockWrapper
            current_underlying_module = self.transformer.module if hasattr(self.transformer, 'module') and isinstance(self.transformer.module, torch.nn.Module) else self.transformer

            if hasattr(self, "first_block_wrapper_for_cache"):
                self.first_block_wrapper_for_cache = FirstBlockWrapper(current_underlying_module)
                self.first_block_hpu_graph_for_cache = wrap_in_hpu_graph(self.first_block_wrapper_for_cache)
            
            # Initialize the main transformer
            htcore.hpu_initialize(self.transformer, mark_only_scales_as_const=True)
        # 다른 모드는 quantize 메서드 사용
        elif quant_mode and not self._has_been_quantized:
            self.quantize(quant_mode=quant_mode)

        # Mixed quantization requires transformer_bf16
        if quant_mode == "quantize-mixed" and not hasattr(self, "transformer_bf16"):
            import copy
            self.transformer_bf16 = copy.deepcopy(self.transformer).to(self._execution_device)

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            num_prompts = 1
        elif prompt is not None and isinstance(prompt, list):
            num_prompts = len(prompt)
        else:
            num_prompts = prompt_embeds.shape[0]
        num_batches = math.ceil((num_images_per_prompt * num_prompts) / batch_size)

        device = self._execution_device

        # 3. Run text encoder
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_image_ids = self.prepare_latents(
            num_prompts * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas,
            mu=mu,
        )
        self._num_timesteps = len(timesteps)

        # handle guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        logger.info(
            f"{num_prompts} prompt(s) received, {num_images_per_prompt} generation(s) per prompt,"
            f" {batch_size} sample(s) per batch, {num_batches} total batch(es)."
        )
        if num_batches < 3:
            logger.warning("The first two iterations are slower so it is recommended to feed more batches.")

        throughput_warmup_steps = kwargs.get("throughput_warmup_steps", 3)
        use_warmup_inference_steps = (
            num_batches <= throughput_warmup_steps and num_inference_steps > throughput_warmup_steps
        )

        ht.hpu.synchronize()
        t0 = time.time()
        t1 = t0

        hb_profiler = HabanaProfile(
            warmup=profiling_warmup_steps,
            active=profiling_steps,
            record_shapes=False,
            # with_stack=True,
        )
        hb_profiler.start()

        # 5.1. Split Input data to batches (HPU-specific step)
        (
            latents_batches,
            text_embeddings_batches,
            pooled_prompt_embeddings_batches,
            guidance_batches,
            num_dummy_samples,
        ) = self._split_inputs_into_batches(batch_size, latents, prompt_embeds, pooled_prompt_embeds, guidance)

        outputs = {
            "images": [],
        }

        # 6. Denoising loop
        for j in range(num_batches):
            # The throughput is calculated from the 4th iteration
            # because compilation occurs in the first 2-3 iterations
            if j == throughput_warmup_steps:
                ht.hpu.synchronize()
                t1 = time.time()

            latents_batch = latents_batches[0]
            latents_batches = torch.roll(latents_batches, shifts=-1, dims=0)
            text_embeddings_batch = text_embeddings_batches[0]
            text_embeddings_batches = torch.roll(text_embeddings_batches, shifts=-1, dims=0)
            pooled_prompt_embeddings_batch = pooled_prompt_embeddings_batches[0]
            pooled_prompt_embeddings_batches = torch.roll(pooled_prompt_embeddings_batches, shifts=-1, dims=0)
            guidance_batch = None if guidance_batches is None else guidance_batches[0]
            guidance_batches = None if guidance_batches is None else torch.roll(guidance_batches, shifts=-1, dims=0)

            if hasattr(self.scheduler, "_init_step_index"):
                # Reset scheduler step index for next batch
                self.scheduler.timesteps = timesteps
                self.scheduler._init_step_index(timesteps[0])

            # Mixed quantization
            quant_mixed_step = len(timesteps)
            if quant_mode == "quantize-mixed":
                # 10% of steps use higher precision in mixed quant mode
                quant_mixed_step = quant_mixed_step - (quant_mixed_step // 10)
                logger.info(f"Use FP8  Transformer at steps 0 to {quant_mixed_step - 1}")
                logger.info(f"Use BF16 Transformer at steps {quant_mixed_step} to {len(timesteps) - 1}")
            htcore.mark_step()
            for i in self.progress_bar(range(len(timesteps))):
                if use_warmup_inference_steps and i == throughput_warmup_steps and j == num_batches - 1:
                    print(f"Synchronizing at step {i}")
                    ht.hpu.synchronize()
                    t1 = time.time()

                if self.interrupt:
                    continue

                timestep = timesteps[0]
                timesteps = torch.roll(timesteps, shifts=-1, dims=0)
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = timestep.expand(latents_batch.shape[0]).to(latents_batch.dtype)

                if i >= quant_mixed_step:
                    # Mixed quantization
                    noise_pred = self.transformer_bf16(
                        hidden_states=latents_batch,
                        timestep=timestep / 1000,
                        guidance=guidance_batch,
                        pooled_projections=pooled_prompt_embeddings_batch,
                        encoder_hidden_states=text_embeddings_batch,
                        txt_ids=text_ids,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                    )[0]
                else:
                    noise_pred = self.transformer(
                        hidden_states=latents_batch,
                        timestep=timestep / 1000,
                        guidance=guidance_batch,
                        pooled_projections=pooled_prompt_embeddings_batch,
                        encoder_hidden_states=text_embeddings_batch,
                        txt_ids=text_ids,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                    )[0]

                # compute the previous noisy sample x_t -> x_t-1
                latents_batch = self.scheduler.step(noise_pred, timestep, latents_batch, return_dict=False)[0]

                hb_profiler.step()
                # htcore.mark_step()
                # htcore.mark_step(sync=True)
                if num_batches > throughput_warmup_steps:
                    ht.hpu.synchronize()

            if not output_type == "latent":
                latents_batch = self._unpack_latents(latents_batch, height, width, self.vae_scale_factor)
                latents_batch = (latents_batch / self.vae.config.scaling_factor) + self.vae.config.shift_factor
                image = self.vae.decode(latents_batch, return_dict=False)[0]
                image = self.image_processor.postprocess(image, output_type=output_type)
            else:
                image = latents_batch

            outputs["images"].append(image)
            # htcore.mark_step()
            # htcore.mark_step(sync=True)

        # 7. Stage after denoising
        hb_profiler.stop()

        if quant_mode == "measure":
            from neural_compressor.torch.quantization import finalize_calibration

            finalize_calibration(self.transformer)
            self._has_been_quantized = True

        # Save FBCache analysis data if enabled
        if ENABLE_FB_ANALYSIS and self.use_hpu_graphs and self.rdt > 0:
            # Check if fb_analysis_data was initialized (it should be if these conditions are met)
            if hasattr(self, "fb_analysis_data") and self.fb_analysis_data["timesteps"]: # Ensure data was collected
                import pickle
                import os
                os.makedirs(output_dir, exist_ok=True)
                analysis_file_path = os.path.join(output_dir, "fb_analysis_data.pkl")
                with open(analysis_file_path, "wb") as f:
                    pickle.dump(self.fb_analysis_data, f)
                logger.info(f"FBCache analysis data saved to {analysis_file_path}")
                # Optionally clear data after saving to free memory for subsequent calls
                # self.fb_analysis_data = {
                #     "timesteps": [],
                #     "first_block_outputs": [],
                #     "total_outputs": [],
                #     "can_use_cache_flags": [],
                #     "first_block_residuals": [],
                # }
            elif hasattr(self, "fb_analysis_data"):
                 logger.info("FBCache analysis was enabled, but no data was collected (e.g., zero inference steps or not using cached path).")
            # else:
            # logger.warning("FBCache analysis enabled, but fb_analysis_data attribute not found. This shouldn't happen if rdt > 0 and use_hpu_graphs.")

        ht.hpu.synchronize()
        speed_metrics_prefix = "generation"
        if use_warmup_inference_steps:
            t1 = warmup_inference_steps_time_adjustment(t1, t1, num_inference_steps, throughput_warmup_steps)
        speed_measures = speed_metrics(
            split=speed_metrics_prefix,
            start_time=t0,
            num_samples=batch_size
            if t1 == t0 or use_warmup_inference_steps
            else (num_batches - throughput_warmup_steps) * batch_size,
            num_steps=batch_size * num_inference_steps
            if use_warmup_inference_steps
            else (num_batches - throughput_warmup_steps) * batch_size * num_inference_steps,
            start_time_after_warmup=t1,
        )
        logger.info(f"Speed metrics: {speed_measures}")

        # 8 Output Images
        if num_dummy_samples > 0:
            # Remove dummy generations if needed
            outputs["images"][-1] = outputs["images"][-1][:-num_dummy_samples]

        # Process generated images
        for i, image in enumerate(outputs["images"][:]):
            if i == 0:
                outputs["images"].clear()

            if output_type == "pil" and isinstance(image, list):
                outputs["images"] += image
            elif output_type in ["np", "numpy"] and isinstance(image, np.ndarray):
                if len(outputs["images"]) == 0:
                    outputs["images"] = image
                else:
                    outputs["images"] = np.concatenate((outputs["images"], image), axis=0)
            else:
                if len(outputs["images"]) == 0:
                    outputs["images"] = image
                else:
                    outputs["images"] = torch.cat((outputs["images"], image), 0)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return outputs["images"]

        return GaudiFluxPipelineOutput(
            images=outputs["images"],
            throughput=speed_measures[f"{speed_metrics_prefix}_samples_per_second"],
        )
