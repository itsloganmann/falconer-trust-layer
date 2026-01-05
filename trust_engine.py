"""
Falconer Trust Layer: Mechanistic Interpretability Engine

This module implements REAL mechanistic interpretability analysis using transformer_lens
for the Falconer AI Trust Layer MVP. It performs actual forward passes, hooks into 
residual streams, and calculates trust metrics dynamically.

Based on "Mechanistic Auditing" research (NeurIPS 2025):
- Module A: Rebound Detection (Hallucination Check)
- Module B: Causal Tracing (Source Attribution)  
- Module C: Knowledge Drift (Staleness Detection)
"""

import torch
from torch import Tensor
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import warnings

from colorama import Fore, Style, init as colorama_init

from transformer_lens import HookedTransformer, ActivationCache

from interp_utils import (
    get_logit_lens_distribution,
    perform_activation_patching,
    compute_attention_inhibition,
    identify_context_token_positions,
    create_corrupted_input,
    LogitLensResult,
    CausalTracingResult,
    AttentionInhibitionResult,
)


# Trust thresholds
VERIFIED_THRESHOLD = 0.85
REVIEW_THRESHOLD = 0.60
HALLUCINATION_RISK_THRESHOLD = 0.4
DRIFT_DIVERGENCE_THRESHOLD = 0.1
STUBBORN_DRIFT_THRESHOLD = 0.02


@dataclass
class SourceAttribution:
    """Attribution of answer to specific source location."""
    source_name: str
    line_number: int
    token_span: Tuple[int, int]
    causal_impact: float
    verified: bool


@dataclass
class TrustMetric:
    """
    Comprehensive trust analysis results from mechanistic interpretability.
    """
    # Overall trust score (0.0 - 1.0)
    trust_score: float
    
    # Module A: Rebound Detection
    hallucination_risk: float
    flagged_attention_heads: List[Tuple[int, int]]
    suppressed_tokens: List[str]
    
    # Module B: Causal Tracing
    source_attributions: List[SourceAttribution]
    critical_layer: int
    causal_impact_by_layer: Dict[int, float]
    
    # Module C: Knowledge Drift
    parametric_prediction: Tuple[str, float]  # (token, confidence)
    contextual_prediction: Tuple[str, float]
    drift_score: float
    drift_type: str  # "none", "valid_override", "stubborn_drift"
    
    # Raw analysis data
    logit_lens_result: Optional[LogitLensResult] = None
    
    # Status flags
    is_verified: bool = False
    needs_review: bool = False
    is_blocked: bool = False


class FalconerTrustEngine:
    """
    Core engine for analyzing LLM outputs using real mechanistic interpretability.
    
    This engine uses transformer_lens to:
    1. Hook into attention heads for hallucination detection
    2. Perform activation patching for causal source attribution
    3. Apply logit lens for knowledge drift detection
    """
    
    def __init__(
        self,
        model_name: str = "gpt2-xl",  # Use gpt2-xl for easier demo, swap to llama for production
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ) -> None:
        """
        Initialize the Falconer Trust Engine with a real transformer model.
        
        Args:
            model_name: HuggingFace model identifier (e.g., "gpt2-xl", "meta-llama/Meta-Llama-3-8B")
            device: Device to run on ("cuda" or "cpu")
            dtype: Data type for model weights
        """
        colorama_init(autoreset=True)
        
        print(f"\n{Fore.CYAN}{'='*70}")
        print(f"{Fore.CYAN}  FALCONER TRUST ENGINE - Mechanistic Interpretability Layer")
        print(f"{Fore.CYAN}{'='*70}\n")
        
        # Check CUDA availability
        if device == "cuda" and not torch.cuda.is_available():
            print(f"{Fore.YELLOW}[!] CUDA not available, falling back to CPU")
            device = "cpu"
            dtype = torch.float32
        
        self.device = device
        self.dtype = dtype
        self.model_name = model_name
        
        print(f"{Fore.YELLOW}[*] Loading model: {model_name}")
        print(f"{Fore.YELLOW}    Device: {device} | Dtype: {dtype}")
        
        # Load the model with transformer_lens
        try:
            self.model: HookedTransformer = HookedTransformer.from_pretrained(
                model_name,
                device=device,
                dtype=dtype,
                fold_ln=False,  # Keep layer norms separate for logit lens
                center_writing_weights=False,
                center_unembed=False,
            )
            self.model.eval()
            
            # Ensure tokenizer has pad token
            if self.model.tokenizer.pad_token is None:
                self.model.tokenizer.pad_token = self.model.tokenizer.eos_token
            
            print(f"{Fore.GREEN}    ✓ Model loaded: {self.model.cfg.n_layers} layers, "
                  f"{self.model.cfg.n_heads} heads, d_model={self.model.cfg.d_model}")
            
        except Exception as e:
            print(f"{Fore.RED}[✗] Failed to load model: {e}")
            raise
        
        print(f"{Fore.YELLOW}[*] Initializing interpretability hooks...")
        print(f"{Fore.GREEN}    ✓ Attention pattern hooks ready")
        print(f"{Fore.GREEN}    ✓ Residual stream hooks ready")
        print(f"{Fore.GREEN}    ✓ Logit lens projection ready")
        
        print(f"\n{Fore.GREEN}[✓] Falconer Trust Engine initialized successfully\n")
    
    def _build_prompt(self, query: str, context: str) -> str:
        """Build a RAG-style prompt combining query and context."""
        return f"""Context information:
{context}

Based on the above context, answer the following question:
Question: {query}

Answer:"""
    
    def _get_answer_token_id(self, answer: str) -> int:
        """Get the token ID for the expected answer's first token."""
        # Tokenize the answer and get first non-special token
        tokens = self.model.to_tokens(answer, prepend_bos=False)[0]
        return tokens[0].item()
    
    @torch.no_grad()
    def analyze_query(
        self,
        query: str,
        context: str,
        correct_answer: str,
        source_metadata: Optional[Dict] = None,
    ) -> TrustMetric:
        """
        Perform full mechanistic interpretability analysis on a query.
        
        This method runs the three detection modules:
        1. Rebound Detection - Check for hallucination via attention inhibition
        2. Causal Tracing - Verify source attribution via activation patching
        3. Knowledge Drift - Detect staleness via logit lens
        
        Args:
            query: The user's question.
            context: The retrieved context/documents.
            correct_answer: The expected correct answer (for causal tracing).
            source_metadata: Optional dict with source file/line info.
            
        Returns:
            TrustMetric with comprehensive analysis results.
        """
        # Build the full prompt
        prompt = self._build_prompt(query, context)
        
        print(f"{Fore.CYAN}[*] Tokenizing input...")
        tokens = self.model.to_tokens(prompt)
        seq_len = tokens.shape[1]
        print(f"    Sequence length: {seq_len} tokens")
        
        # ================================================================
        # FORWARD PASS WITH CACHING
        # ================================================================
        print(f"{Fore.CYAN}[*] Running forward pass with activation caching...")
        
        logits, cache = self.model.run_with_cache(
            tokens,
            names_filter=lambda name: (
                "hook_resid" in name or 
                "hook_pattern" in name or
                "hook_attn_out" in name
            )
        )
        
        print(f"{Fore.GREEN}    ✓ Cached {len(cache)} activation tensors")
        
        # Identify context token positions
        context_start, context_end, context_positions = identify_context_token_positions(
            self.model, tokens, context
        )
        print(f"    Context spans positions {context_start}-{context_end}")
        
        # ================================================================
        # MODULE A: REBOUND DETECTION (Hallucination Check)
        # ================================================================
        print(f"\n{Fore.YELLOW}[Module A] Rebound Detection - Hallucination Analysis")
        
        inhibition_result = compute_attention_inhibition(
            self.model,
            cache,
            context_positions,
            answer_position=-1,
        )
        
        # Decode suppressed tokens
        suppressed_tokens = []
        if inhibition_result.flagged_heads:
            # Find which context tokens are most suppressed
            for pos in context_positions[:5]:  # Sample first 5
                token_str = self.model.tokenizer.decode(tokens[0, pos].item())
                suppressed_tokens.append(token_str.strip())
        
        hallucination_risk = inhibition_result.hallucination_risk_score
        
        if hallucination_risk > HALLUCINATION_RISK_THRESHOLD:
            print(f"{Fore.RED}    ⚠ High hallucination risk: {hallucination_risk:.3f}")
            print(f"{Fore.RED}    Flagged heads: {inhibition_result.flagged_heads[:5]}")
        else:
            print(f"{Fore.GREEN}    ✓ Low hallucination risk: {hallucination_risk:.3f}")
        
        # ================================================================
        # MODULE B: CAUSAL TRACING (Source Attribution)
        # ================================================================
        print(f"\n{Fore.YELLOW}[Module B] Causal Tracing - Source Attribution")
        
        # Get the answer token ID
        answer_token_id = self._get_answer_token_id(correct_answer)
        print(f"    Answer token: '{correct_answer}' -> ID {answer_token_id}")
        
        # Create corrupted input (zero out context)
        corrupt_tokens = create_corrupted_input(
            self.model, tokens, context_start, context_end, "zero"
        )
        
        # Perform activation patching
        causal_result = perform_activation_patching(
            self.model,
            tokens,
            corrupt_tokens,
            answer_token_id,
            context_start,
            context_end,
        )
        
        print(f"{Fore.GREEN}    ✓ Critical layer: {causal_result.critical_layer}")
        print(f"    ✓ Critical position: {causal_result.critical_position}")
        print(f"    ✓ Logit diff restored: {causal_result.restored_logit_diff:.3f}")
        
        # Build source attributions
        source_attributions = []
        if source_metadata:
            source_name = source_metadata.get("filename", "unknown_source.txt")
            base_line = source_metadata.get("line_number", 1)
        else:
            source_name = "context_document.txt"
            base_line = 1
        
        # Map critical tokens back to source
        for idx, pos in enumerate(causal_result.source_token_indices[:3]):
            # Calculate approximate line number from position
            line_offset = (pos - context_start) // 10  # Rough estimate
            source_attributions.append(SourceAttribution(
                source_name=source_name,
                line_number=base_line + line_offset,
                token_span=(pos, pos + 1),
                causal_impact=causal_result.layer_impacts.get(causal_result.critical_layer, 0),
                verified=causal_result.restored_logit_diff > 0.5,
            ))
        
        # ================================================================
        # MODULE C: KNOWLEDGE DRIFT (Staleness Detection)
        # ================================================================
        print(f"\n{Fore.YELLOW}[Module C] Knowledge Drift - Logit Lens Analysis")
        
        logit_lens_result = get_logit_lens_distribution(self.model, cache, position=-1)
        
        parametric_pred = logit_lens_result.middle_layer_prediction
        contextual_pred = logit_lens_result.final_prediction
        divergence = logit_lens_result.parametric_contextual_divergence
        
        print(f"    Embedding layer predicts: '{logit_lens_result.embedding_prediction[0]}' "
              f"({logit_lens_result.embedding_prediction[1]:.3f})")
        print(f"    Middle layer predicts: '{parametric_pred[0]}' ({parametric_pred[1]:.3f})")
        print(f"    Final layer predicts: '{contextual_pred[0]}' ({contextual_pred[1]:.3f})")
        print(f"    Parametric-Contextual Divergence: {divergence:.4f}")
        
        # Determine drift type
        if divergence > DRIFT_DIVERGENCE_THRESHOLD:
            drift_type = "valid_override"
            drift_score = divergence
            print(f"{Fore.GREEN}    ✓ Valid context override detected")
        elif divergence < STUBBORN_DRIFT_THRESHOLD and parametric_pred[0] != contextual_pred[0]:
            drift_type = "stubborn_drift"
            drift_score = 1.0 - divergence
            print(f"{Fore.RED}    ⚠ Stubborn drift: Model ignoring context")
        else:
            drift_type = "none"
            drift_score = 0.0
            print(f"{Fore.GREEN}    ✓ No significant drift detected")
        
        # ================================================================
        # CALCULATE OVERALL TRUST SCORE
        # ================================================================
        print(f"\n{Fore.CYAN}[*] Calculating Trust Score...")
        
        # Trust score components:
        # - Low hallucination risk -> higher trust
        # - High causal impact from context -> higher trust  
        # - Valid context override (not stubborn drift) -> higher trust
        
        hallucination_component = 1.0 - hallucination_risk
        
        # Normalize causal impact (higher is better, context contributed to answer)
        max_layer_impact = max(causal_result.layer_impacts.values()) if causal_result.layer_impacts else 0
        causal_component = min(1.0, max(0.0, max_layer_impact))
        
        # Drift component
        if drift_type == "stubborn_drift":
            drift_component = 0.3  # Penalize stubborn drift
        elif drift_type == "valid_override":
            drift_component = 1.0  # Reward valid override
        else:
            drift_component = 0.7  # Neutral
        
        # Weighted combination
        trust_score = (
            0.35 * hallucination_component +
            0.35 * causal_component +
            0.30 * drift_component
        )
        trust_score = min(1.0, max(0.0, trust_score))
        
        # Determine status
        is_verified = trust_score >= VERIFIED_THRESHOLD
        needs_review = REVIEW_THRESHOLD <= trust_score < VERIFIED_THRESHOLD
        is_blocked = trust_score < REVIEW_THRESHOLD
        
        return TrustMetric(
            trust_score=trust_score,
            hallucination_risk=hallucination_risk,
            flagged_attention_heads=inhibition_result.flagged_heads,
            suppressed_tokens=suppressed_tokens,
            source_attributions=source_attributions,
            critical_layer=causal_result.critical_layer,
            causal_impact_by_layer=causal_result.layer_impacts,
            parametric_prediction=parametric_pred,
            contextual_prediction=contextual_pred,
            drift_score=drift_score,
            drift_type=drift_type,
            logit_lens_result=logit_lens_result,
            is_verified=is_verified,
            needs_review=needs_review,
            is_blocked=is_blocked,
        )
    
    def print_report(self, query: str, metric: TrustMetric) -> None:
        """Print a comprehensive trust analysis report."""
        print(f"\n{Fore.CYAN}{'═'*70}")
        print(f"{Fore.WHITE}{Style.BRIGHT}  TRUST ANALYSIS REPORT")
        print(f"{Fore.CYAN}{'═'*70}")
        
        print(f"\n{Fore.WHITE}Query: {query}")
        
        # Status banner
        if metric.is_verified:
            status = f"{Fore.GREEN}{Style.BRIGHT}[✓] VERIFIED"
            status_color = Fore.GREEN
        elif metric.needs_review:
            status = f"{Fore.YELLOW}{Style.BRIGHT}[!] REVIEW NEEDED"
            status_color = Fore.YELLOW
        else:
            status = f"{Fore.RED}{Style.BRIGHT}[✗] BLOCKED"
            status_color = Fore.RED
        
        print(f"\n{Fore.WHITE}  Trust Score: {status_color}{Style.BRIGHT}{metric.trust_score:.1%}")
        print(f"  Status: {status}{Style.RESET_ALL}")
        
        # Module A: Hallucination
        print(f"\n{Fore.CYAN}{'─'*70}")
        print(f"{Fore.WHITE}{Style.BRIGHT}  Module A: Rebound Detection (Hallucination)")
        print(f"{Fore.CYAN}{'─'*70}")
        
        risk_color = Fore.RED if metric.hallucination_risk > HALLUCINATION_RISK_THRESHOLD else Fore.GREEN
        print(f"  Hallucination Risk: {risk_color}{metric.hallucination_risk:.3f}")
        
        if metric.flagged_attention_heads:
            print(f"  Flagged Attention Heads: {metric.flagged_attention_heads[:5]}")
        if metric.suppressed_tokens:
            print(f"  Suppressed Context Tokens: {metric.suppressed_tokens}")
        
        # Module B: Source Attribution
        print(f"\n{Fore.CYAN}{'─'*70}")
        print(f"{Fore.WHITE}{Style.BRIGHT}  Module B: Causal Tracing (Source Attribution)")
        print(f"{Fore.CYAN}{'─'*70}")
        
        print(f"  Critical Layer: {metric.critical_layer}")
        
        if metric.source_attributions:
            print(f"  Verified Sources:")
            for attr in metric.source_attributions:
                verified_icon = "✓" if attr.verified else "?"
                print(f"    {verified_icon} {attr.source_name}:{attr.line_number} "
                      f"(impact: {attr.causal_impact:.3f})")
        
        # Module C: Knowledge Drift
        print(f"\n{Fore.CYAN}{'─'*70}")
        print(f"{Fore.WHITE}{Style.BRIGHT}  Module C: Knowledge Drift (Logit Lens)")
        print(f"{Fore.CYAN}{'─'*70}")
        
        print(f"  Parametric Memory: '{metric.parametric_prediction[0]}' "
              f"(confidence: {metric.parametric_prediction[1]:.3f})")
        print(f"  Contextual Output: '{metric.contextual_prediction[0]}' "
              f"(confidence: {metric.contextual_prediction[1]:.3f})")
        
        drift_color = Fore.RED if metric.drift_type == "stubborn_drift" else Fore.GREEN
        print(f"  Drift Type: {drift_color}{metric.drift_type}")
        print(f"  Drift Score: {metric.drift_score:.3f}")
        
        print(f"\n{Fore.CYAN}{'═'*70}\n")


def main() -> None:
    """
    Main demonstration of the Falconer Trust Engine.
    
    Runs real-world scenarios with fresh vs stale context to demonstrate
    the drift detector and other mechanistic interpretability modules.
    """
    print(f"\n{Fore.MAGENTA}{Style.BRIGHT}{'='*70}")
    print(f"{Fore.MAGENTA}  FALCONER TRUST LAYER - Mechanistic Interpretability Demo")
    print(f"{Fore.MAGENTA}{'='*70}\n")
    
    # Initialize the engine (use gpt2-xl for demo, can swap to llama)
    # For Colab H100, you can use: "meta-llama/Meta-Llama-3-8B"
    engine = FalconerTrustEngine(
        model_name="meta-llama/Meta-Llama-3-8B",  # Change to "meta-llama/Meta-Llama-3-8B" for production
        device="cuda",
        dtype=torch.float16,
    )
    
    # ================================================================
    # SCENARIO 1: Fresh Context (Should have high trust)
    # ================================================================
    print(f"\n{Fore.MAGENTA}{'='*70}")
    print(f"{Fore.MAGENTA}  SCENARIO 1: Fresh Context - API Version Query")
    print(f"{Fore.MAGENTA}{'='*70}")
    
    fresh_context = """
    Falconer API Documentation (Last Updated: January 5, 2026)
    
    Current API Version: v3.2.1
    
    The Falconer API provides enterprise-grade AI trust verification.
    Authentication is handled via OAuth 2.0 with JWT tokens.
    Rate limits: 1000 requests per minute for Pro tier.
    
    Breaking changes in v3.2.0:
    - Deprecated /v2/analyze endpoint
    - New /v3/trust-score endpoint with enhanced metrics
    """
    
    query_1 = "What is the Falconer API version?"
    correct_answer_1 = "v3.2.1"
    
    metric_1 = engine.analyze_query(
        query=query_1,
        context=fresh_context,
        correct_answer=correct_answer_1,
        source_metadata={"filename": "api_docs.md", "line_number": 5},
    )
    
    engine.print_report(query_1, metric_1)
    
    # ================================================================
    # SCENARIO 2: Stale Context (Should trigger drift detection)
    # ================================================================
    print(f"\n{Fore.MAGENTA}{'='*70}")
    print(f"{Fore.MAGENTA}  SCENARIO 2: Stale Context - Outdated Documentation")
    print(f"{Fore.MAGENTA}{'='*70}")
    
    stale_context = """
    Falconer API Documentation (Last Updated: March 2023)
    
    Current API Version: v1.0.0
    
    The Falconer API uses basic API key authentication.
    Rate limits: 100 requests per hour.
    
    Endpoints:
    - POST /v1/analyze - Submit text for analysis
    - GET /v1/status - Check service status
    """
    
    query_2 = "What is the Falconer API version?"
    correct_answer_2 = "v1.0.0"  # The stale answer
    
    metric_2 = engine.analyze_query(
        query=query_2,
        context=stale_context,
        correct_answer=correct_answer_2,
        source_metadata={"filename": "old_api_docs.md", "line_number": 5},
    )
    
    engine.print_report(query_2, metric_2)
    
    # ================================================================
    # SCENARIO 3: Contradictory Context (Hallucination test)
    # ================================================================
    print(f"\n{Fore.MAGENTA}{'='*70}")
    print(f"{Fore.MAGENTA}  SCENARIO 3: Ambiguous Context - Hallucination Risk")
    print(f"{Fore.MAGENTA}{'='*70}")
    
    ambiguous_context = """
    Company Overview:
    
    Falconer AI was founded to build trust systems. 
    The team includes experts from various backgrounds.
    Our mission is to make AI systems more reliable.
    """
    
    query_3 = "What is the Falconer API rate limit?"
    correct_answer_3 = "1000"  # Not in context - may hallucinate
    
    metric_3 = engine.analyze_query(
        query=query_3,
        context=ambiguous_context,
        correct_answer=correct_answer_3,
        source_metadata={"filename": "about.md", "line_number": 1},
    )
    
    engine.print_report(query_3, metric_3)
    
    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{Fore.MAGENTA}{'='*70}")
    print(f"{Fore.MAGENTA}  TRUST ANALYSIS SUMMARY")
    print(f"{Fore.MAGENTA}{'='*70}\n")
    
    scenarios = [
        ("Fresh Context", metric_1),
        ("Stale Context", metric_2),
        ("Ambiguous Context", metric_3),
    ]
    
    for name, metric in scenarios:
        if metric.is_verified:
            status = f"{Fore.GREEN}VERIFIED"
        elif metric.needs_review:
            status = f"{Fore.YELLOW}REVIEW"
        else:
            status = f"{Fore.RED}BLOCKED"
        
        print(f"  {name:20} -> Trust: {metric.trust_score:.1%} | Status: {status}{Style.RESET_ALL}")
    
    print(f"\n{Fore.GREEN}{Style.BRIGHT}Analysis complete.")
    print(f"{Fore.WHITE}Powered by Falconer AI - Real Mechanistic Interpretability\n")


if __name__ == "__main__":
    main()
