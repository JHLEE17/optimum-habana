from transformers.generation import GenerationConfig


class GaudiGenerationConfig(GenerationConfig):
    """
    This class extends [`transformers.generation.GenerationConfig`](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/configuration_utils.py)
    to add HPU-specific arguments for generation.

    Arg:
    trim_logit (`bool`, *optional):
        Calculate logits only for the last token to save memory in the first step.
    static_shapes (`bool`, *optional*):
        Whether to use static shapes for generation or not. It will run faster on HPUs with static shapes
        but not all models support it. If not specified, it will automatically be set to `True` if the given
        model supports it.
    ignore_eos (`bool`, *optional*):
        Whether to ignore finished sequences (faster in lazy mode and with HPU graphs) or not (eager mode).
        If not specified, it will automatically be set to `True` if lazy mode is on.
    attn_softmax_bf16 (`bool`, *optional*):
        Whether to run attention softmax layer in lower precision provided that the model supports it and
        is also running in lower precision.
    limit_hpu_graphs (`bool`, *optional*):
        Skip HPU Graph usage for first token to save memory
    clear_hpu_graphs_cache (`bool`, *optional*):
        Clear HPU Graph cache
    reuse_cache (`bool`, *optional*):
        Whether to reuse key/value cache for decoding. It should save memory.
    bucket_size (`int`, *optional*):
        If negative (default=-1) pad to max if `static_shapes` is set. Else start with
        `shape = bucket_size * ceil(prompt_len/bucket_size)` and then grow space by `bucket_size` when needed.
        Only active if `static_shapes` is used. Can't be used with `reuse_cache`.
    bucket_internal (`bool`, *optional*):
        Split kv sequence into buckets in decode phase. It improves throughput when max_new_tokens is large.
    use_flash_attention (`bool`, *optional*):
        Whether to use flash attention optimization.
    flash_attention_recompute (`bool`, *optional*):
        Whether to enable recompute if use Habana flash attention.
    flash_attention_causal_mask (`bool`, *optional*):
        Whether to enable causal_mask if use Habana flash attention.
    flash_attention_fast_softmax_mode (`bool`, *optional*):
        Whether to use fast softmax with reduced precision if use Habana flash attention.
    attn_batch_split (`int`, *optional*):
        Specify the batch size split for attention and mlp layers. 1 for no split. This is enabled only for prompt.
    logits_bf16 (`bool`, *optional*):
        Keep logits in bf16.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.trim_logits = kwargs.get("trim_logits", None)
        self.static_shapes = kwargs.get("static_shapes", None)
        self.ignore_eos = kwargs.get("ignore_eos", None)
        self.attn_softmax_bf16 = kwargs.get("attn_softmax_bf16", None)
        self.limit_hpu_graphs = kwargs.get("limit_hpu_graphs", None)
        self.clear_hpu_graphs_cache = kwargs.get("clear_hpu_graphs_cache", None)
        self.reuse_cache = kwargs.get("reuse_cache", None)
        self.bucket_size = kwargs.get("bucket_size", -1)
        self.bucket_internal = kwargs.get("bucket_internal", None)
        self.reduce_recompile = kwargs.get("reduce_recompile", None)
        self.use_flash_attention = kwargs.get("use_flash_attention", None)
        self.flash_attention_recompute = kwargs.get("flash_attention_recompute", None)
        self.flash_attention_causal_mask = kwargs.get("flash_attention_causal_mask", None)
        self.flash_attention_fast_softmax = kwargs.get("flash_attention_fast_softmax", None)
        self.use_fused_rope = kwargs.get("use_fused_rope", None)
        self.valid_sequence_lengths = kwargs.get("valid_sequence_lengths", None)
        self.attn_batch_split = kwargs.get("attn_batch_split", 1)
        self.logits_bf16 = kwargs.get("logits_bf16", None)
