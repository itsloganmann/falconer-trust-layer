"""
Falconer Trust Layer: Interpretability Utilities

This module provides the core mechanistic interpretability functions using
transformer_lens for real forward passes, activation patching, and logit lens.
"""

import torch
from torch import Tensor
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import einops
from jaxtyping import Float, Int

from transformer_lens import HookedTransformer, ActivationCache
from transformer_lens.hook_points import HookPoint


@dataclass
class LogitLensResult:
    """Results from logit lens analysis at different layers."""
    layer_predictions: Dict[int, Tuple[str, float]]  # layer -> (top_token, probability)
    embedding_prediction: Tuple[str, float]
    middle_layer_prediction: Tuple[str, float]
    final_prediction: Tuple[str, float]
    parametric_contextual_divergence: float  # How much middle differs from final
    

@dataclass
class CausalTracingResult:
    """Results from activation patching / causal tracing."""
    layer_impacts: Dict[int, float]  # layer -> causal impact score
    critical_layer: int
    critical_position: int
    source_token_indices: List[int]
    restored_logit_diff: float


@dataclass
class AttentionInhibitionResult:
    """Results from attention head inhibition analysis (Rebound Detection)."""
    head_inhibition_scores: Dict[Tuple[int, int], float]  # (layer, head) -> inhibition score
    suppressed_context_tokens: List[str]
    hallucination_risk_score: float
    flagged_heads: List[Tuple[int, int]]


def get_logit_lens_distribution(
    model: HookedTransformer,
    cache: ActivationCache,
    position: int = -1,
) -> LogitLensResult:
    """
    Apply the Logit Lens technique to decode residual stream at each layer.
    
    Projects intermediate activations through the unembedding matrix to see
    what the model would predict at each layer.
    
    Args:
        model: The HookedTransformer model.
        cache: Activation cache from a forward pass.
        position: Token position to analyze (-1 for last token).
        
    Returns:
        LogitLensResult with predictions at each layer.
    """
    n_layers = model.cfg.n_layers
    layer_predictions: Dict[int, Tuple[str, float]] = {}
    
    # Get the unembedding matrix and layer norm
    W_U = model.W_U  # [d_model, d_vocab]
    
    # Analyze each layer's residual stream
    for layer in range(n_layers):
        # Get residual stream at this layer (after the layer's computation)
        resid_key = f"blocks.{layer}.hook_resid_post"
        if resid_key not in cache:
            resid_key = f"blocks.{layer}.hook_resid_pre"
        
        resid: Float[Tensor, "batch seq d_model"] = cache[resid_key]
        
        # Apply final layer norm before unembedding
        resid_normed = model.ln_final(resid[:, position, :])
        
        # Project to vocabulary space
        logits: Float[Tensor, "batch d_vocab"] = resid_normed @ W_U
        
        # Get probabilities and top prediction
        probs = torch.softmax(logits, dim=-1)
        top_prob, top_idx = probs.max(dim=-1)
        top_token = model.tokenizer.decode(top_idx.item())
        
        layer_predictions[layer] = (top_token, top_prob.item())
    
    # Get embedding layer prediction (layer 0 residual)
    embed_resid = cache["blocks.0.hook_resid_pre"][:, position, :]
    embed_normed = model.ln_final(embed_resid)
    embed_logits = embed_normed @ W_U
    embed_probs = torch.softmax(embed_logits, dim=-1)
    embed_top_prob, embed_top_idx = embed_probs.max(dim=-1)
    embedding_prediction = (model.tokenizer.decode(embed_top_idx.item()), embed_top_prob.item())
    
    # Middle layer prediction
    middle_layer = n_layers // 2
    middle_prediction = layer_predictions[middle_layer]
    
    # Final layer prediction
    final_prediction = layer_predictions[n_layers - 1]
    
    # Calculate divergence between parametric (middle) and contextual (final)
    # High divergence = context is overriding parametric memory
    # Low divergence with different context = stubborn drift
    middle_resid = cache[f"blocks.{middle_layer}.hook_resid_post"][:, position, :]
    middle_normed = model.ln_final(middle_resid)
    middle_logits = middle_normed @ W_U
    
    final_resid = cache[f"blocks.{n_layers-1}.hook_resid_post"][:, position, :]
    final_normed = model.ln_final(final_resid)
    final_logits = final_normed @ W_U
    
    # KL divergence between middle and final distributions
    middle_probs = torch.softmax(middle_logits, dim=-1)
    final_probs = torch.softmax(final_logits, dim=-1)
    
    # Use Jensen-Shannon divergence for symmetry
    m_probs = 0.5 * (middle_probs + final_probs)
    kl_mid = torch.sum(middle_probs * (torch.log(middle_probs + 1e-10) - torch.log(m_probs + 1e-10)))
    kl_fin = torch.sum(final_probs * (torch.log(final_probs + 1e-10) - torch.log(m_probs + 1e-10)))
    js_divergence = 0.5 * (kl_mid + kl_fin)
    
    return LogitLensResult(
        layer_predictions=layer_predictions,
        embedding_prediction=embedding_prediction,
        middle_layer_prediction=middle_prediction,
        final_prediction=final_prediction,
        parametric_contextual_divergence=js_divergence.item(),
    )


def perform_activation_patching(
    model: HookedTransformer,
    clean_tokens: Int[Tensor, "batch seq"],
    corrupt_tokens: Int[Tensor, "batch seq"],
    answer_token_idx: int,
    context_start_pos: int,
    context_end_pos: int,
) -> CausalTracingResult:
    """
    Perform activation patching (denoising) to trace causal impact.
    
    Runs clean and corrupted passes, then patches activations from clean
    into corrupted to measure which positions/layers recover the answer.
    
    Args:
        model: The HookedTransformer model.
        clean_tokens: Tokenized input with correct context.
        corrupt_tokens: Tokenized input with zeroed/corrupted context.
        answer_token_idx: The token index of the expected answer in vocabulary.
        context_start_pos: Start position of context tokens.
        context_end_pos: End position of context tokens.
        
    Returns:
        CausalTracingResult with causal impact analysis.
    """
    n_layers = model.cfg.n_layers
    
    # Run clean forward pass and cache
    with torch.no_grad():
        clean_logits, clean_cache = model.run_with_cache(clean_tokens)
    
    # Get clean answer logit at final position
    clean_answer_logit = clean_logits[0, -1, answer_token_idx].item()
    
    # Run corrupted forward pass
    with torch.no_grad():
        corrupt_logits, corrupt_cache = model.run_with_cache(corrupt_tokens)
    
    corrupt_answer_logit = corrupt_logits[0, -1, answer_token_idx].item()
    
    # Baseline logit difference
    baseline_diff = clean_answer_logit - corrupt_answer_logit
    
    layer_impacts: Dict[int, float] = {}
    position_impacts: Dict[int, Dict[int, float]] = {}  # layer -> {position -> impact}
    
    # Patch each layer's residual stream at context positions
    for layer in range(n_layers):
        position_impacts[layer] = {}
        layer_total_impact = 0.0
        
        for pos in range(context_start_pos, min(context_end_pos, clean_tokens.shape[1])):
            def patch_hook(
                activations: Float[Tensor, "batch seq d_model"],
                hook: HookPoint,
                clean_act: Float[Tensor, "batch seq d_model"] = clean_cache[f"blocks.{layer}.hook_resid_post"],
                patch_pos: int = pos,
            ) -> Float[Tensor, "batch seq d_model"]:
                # Patch in the clean activation at this position
                activations[:, patch_pos, :] = clean_act[:, patch_pos, :]
                return activations
            
            # Run corrupted with patch
            with torch.no_grad():
                patched_logits = model.run_with_hooks(
                    corrupt_tokens,
                    fwd_hooks=[(f"blocks.{layer}.hook_resid_post", patch_hook)],
                )
            
            patched_answer_logit = patched_logits[0, -1, answer_token_idx].item()
            
            # Calculate restoration (how much patching recovered the clean behavior)
            restoration = patched_answer_logit - corrupt_answer_logit
            normalized_impact = restoration / (baseline_diff + 1e-10)
            
            position_impacts[layer][pos] = normalized_impact
            layer_total_impact += normalized_impact
        
        layer_impacts[layer] = layer_total_impact / max(1, context_end_pos - context_start_pos)
    
    # Find critical layer and position
    critical_layer = max(layer_impacts.keys(), key=lambda k: layer_impacts[k])
    critical_position = context_start_pos
    max_pos_impact = 0.0
    
    for pos, impact in position_impacts[critical_layer].items():
        if impact > max_pos_impact:
            max_pos_impact = impact
            critical_position = pos
    
    # Identify source token indices (positions with high causal impact)
    source_tokens = []
    for pos in range(context_start_pos, context_end_pos):
        avg_impact = sum(position_impacts[l].get(pos, 0) for l in range(n_layers)) / n_layers
        if avg_impact > 0.1:  # Threshold for significance
            source_tokens.append(pos)
    
    return CausalTracingResult(
        layer_impacts=layer_impacts,
        critical_layer=critical_layer,
        critical_position=critical_position,
        source_token_indices=source_tokens if source_tokens else [critical_position],
        restored_logit_diff=baseline_diff,
    )


def compute_attention_inhibition(
    model: HookedTransformer,
    cache: ActivationCache,
    context_token_positions: List[int],
    answer_position: int = -1,
) -> AttentionInhibitionResult:
    """
    Compute attention head inhibition scores for Rebound Detection.
    
    Analyzes which attention heads actively suppress (down-weight) tokens
    from the context, which may indicate hallucination risk.
    
    Args:
        model: The HookedTransformer model.
        cache: Activation cache from forward pass.
        context_token_positions: Positions of context tokens to monitor.
        answer_position: Position of the answer token being generated.
        
    Returns:
        AttentionInhibitionResult with inhibition analysis.
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    
    head_inhibition_scores: Dict[Tuple[int, int], float] = {}
    flagged_heads: List[Tuple[int, int]] = []
    
    # Analyze attention patterns at each layer/head
    for layer in range(n_layers):
        # Get attention pattern: [batch, head, query_pos, key_pos]
        attn_pattern = cache[f"blocks.{layer}.attn.hook_pattern"]
        
        for head in range(n_heads):
            # Get attention from answer position to context positions
            if answer_position == -1:
                answer_idx = attn_pattern.shape[2] - 1
            else:
                answer_idx = answer_position
            
            # Attention weights from answer to context tokens
            context_attention = attn_pattern[0, head, answer_idx, context_token_positions]
            
            # Calculate inhibition: if attention to context is very low relative to expected
            # uniform attention, this head may be suppressing context
            expected_attention = 1.0 / attn_pattern.shape[3]  # Uniform baseline
            avg_context_attention = context_attention.mean().item()
            
            # Inhibition score: negative means suppression
            # (expected - actual) / expected, scaled
            inhibition = (expected_attention - avg_context_attention) / (expected_attention + 1e-10)
            
            head_inhibition_scores[(layer, head)] = inhibition
            
            # Flag heads with significant negative inhibition (active suppression)
            if inhibition > 0.5:  # Head is giving much less attention to context than expected
                flagged_heads.append((layer, head))
    
    # Calculate overall hallucination risk
    avg_inhibition = sum(head_inhibition_scores.values()) / len(head_inhibition_scores)
    n_flagged = len(flagged_heads)
    
    # Hallucination risk: combination of avg inhibition and number of flagged heads
    hallucination_risk = min(1.0, max(0.0, 
        0.5 * avg_inhibition + 0.5 * (n_flagged / (n_layers * n_heads / 4))
    ))
    
    return AttentionInhibitionResult(
        head_inhibition_scores=head_inhibition_scores,
        suppressed_context_tokens=[],  # Will be filled by caller with decoded tokens
        hallucination_risk_score=hallucination_risk,
        flagged_heads=flagged_heads,
    )


def identify_context_token_positions(
    model: HookedTransformer,
    full_tokens: Int[Tensor, "batch seq"],
    context_text: str,
) -> Tuple[int, int, List[int]]:
    """
    Identify the token positions corresponding to the context in the full input.
    
    Args:
        model: The HookedTransformer model.
        full_tokens: The full tokenized input.
        context_text: The context string to locate.
        
    Returns:
        Tuple of (start_pos, end_pos, list of positions).
    """
    # Tokenize just the context
    context_tokens = model.to_tokens(context_text, prepend_bos=False)[0]
    full_seq = full_tokens[0]
    
    # Sliding window search for context tokens in full sequence
    context_len = len(context_tokens)
    
    for start in range(len(full_seq) - context_len + 1):
        if torch.equal(full_seq[start:start + context_len], context_tokens):
            positions = list(range(start, start + context_len))
            return start, start + context_len, positions
    
    # Fallback: estimate based on position (context usually comes after query)
    # Assume context starts around 1/4 into the sequence
    estimated_start = len(full_seq) // 4
    estimated_end = 3 * len(full_seq) // 4
    positions = list(range(estimated_start, estimated_end))
    
    return estimated_start, estimated_end, positions


def create_corrupted_input(
    model: HookedTransformer,
    clean_tokens: Int[Tensor, "batch seq"],
    context_start: int,
    context_end: int,
    corruption_type: str = "zero",
) -> Int[Tensor, "batch seq"]:
    """
    Create corrupted version of input for causal tracing.
    
    Args:
        model: The HookedTransformer model.
        clean_tokens: Original clean tokens.
        context_start: Start position of context to corrupt.
        context_end: End position of context to corrupt.
        corruption_type: Type of corruption ("zero", "random", "pad").
        
    Returns:
        Corrupted tokens tensor.
    """
    corrupt_tokens = clean_tokens.clone()
    
    if corruption_type == "zero":
        # Replace context with padding token (or a neutral token)
        pad_token = model.tokenizer.pad_token_id or 0
        corrupt_tokens[0, context_start:context_end] = pad_token
    elif corruption_type == "random":
        # Replace with random tokens from vocabulary
        vocab_size = model.cfg.d_vocab
        random_tokens = torch.randint(0, vocab_size, (context_end - context_start,))
        corrupt_tokens[0, context_start:context_end] = random_tokens.to(corrupt_tokens.device)
    elif corruption_type == "pad":
        # Use specific padding
        pad_token = model.tokenizer.pad_token_id or model.tokenizer.eos_token_id or 0
        corrupt_tokens[0, context_start:context_end] = pad_token
    
    return corrupt_tokens
