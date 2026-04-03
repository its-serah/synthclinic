"""
synthclinic.evaluation.clinical
----------------------------
Domain-specific clinical validity metrics.

ECG validity
  R-peak detection rate  : fraction of synthetic segments with detectable R-peaks
  QRS duration           : distribution comparison (synthetic vs real)
  RR interval statistics : mean ± std of inter-beat interval
  Signal-to-Noise Ratio  : proxy for signal quality

Clinical notes
  Perplexity             : GPT-2 log-perplexity of generated text (lower = more fluent)
  BLEU-2 / ROUGE-L       : n-gram overlap vs reference corpus
  Vocabulary coverage    : fraction of medical terms present in generated notes

These are heuristic but widely used in the medical generative AI literature.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ECG clinical metrics
# ---------------------------------------------------------------------------

def ecg_morphology_metrics(
    signals: np.ndarray,
    fs: float = 360.0,
) -> Dict[str, float]:
    """
    Compute morphology statistics on synthetic ECG segments.

    Parameters
    ----------
    signals : (N, T) float array — synthetic ECG segments
    fs      : sampling frequency (Hz)

    Returns
    -------
    dict with:
        ``"r_peak_detection_rate"`` : fraction with ≥ 1 detected R-peak
        ``"mean_rr_interval_ms"``   : mean RR interval in milliseconds
        ``"std_rr_interval_ms"``    : std of RR intervals
        ``"mean_qrs_duration_ms"``  : estimated mean QRS duration
        ``"mean_snr_db"``           : mean signal-to-noise ratio
    """
    try:
        import scipy.signal as ssig
    except ImportError:
        logger.warning("scipy not available for ECG metrics")
        return {}

    if signals.ndim == 3:
        signals = signals[:, :, 0]  # (N, T, 1) → (N, T)

    r_detected = 0
    rr_intervals_ms = []
    qrs_durations_ms = []
    snrs_db = []

    for seg in signals:
        # Detect R-peaks using Pan-Tompkins-like prominence detection
        try:
            peaks, props = ssig.find_peaks(
                seg,
                height=np.mean(seg) + 0.3 * np.std(seg),
                distance=int(0.25 * fs),  # min 250 ms between beats
                prominence=0.2,
            )
        except Exception:
            peaks = np.array([])

        if len(peaks) >= 1:
            r_detected += 1

        if len(peaks) >= 2:
            rr_ms = np.diff(peaks) / fs * 1000
            rr_intervals_ms.extend(rr_ms.tolist())

        # QRS proxy: width of the dominant peak at half-prominence
        if len(peaks) > 0 and "widths" not in props:
            try:
                widths, *_ = ssig.peak_widths(seg, peaks, rel_height=0.5)
                qrs_durations_ms.extend((widths / fs * 1000).tolist())
            except Exception:
                pass

        # SNR: ratio of signal power to noise power
        # Noise estimated as high-frequency residual after lowpass filter
        try:
            b, a = ssig.butter(4, 40.0 / (fs / 2), btype="low")
            filtered = ssig.filtfilt(b, a, seg)
            noise = seg - filtered
            signal_power = np.mean(filtered ** 2) + 1e-12
            noise_power = np.mean(noise ** 2) + 1e-12
            snr = 10 * np.log10(signal_power / noise_power)
            snrs_db.append(snr)
        except Exception:
            pass

    n = len(signals)
    results: Dict[str, float] = {
        "r_peak_detection_rate": r_detected / n if n > 0 else float("nan"),
        "mean_rr_interval_ms": float(np.mean(rr_intervals_ms)) if rr_intervals_ms else float("nan"),
        "std_rr_interval_ms": float(np.std(rr_intervals_ms)) if rr_intervals_ms else float("nan"),
        "mean_qrs_duration_ms": float(np.mean(qrs_durations_ms)) if qrs_durations_ms else float("nan"),
        "mean_snr_db": float(np.mean(snrs_db)) if snrs_db else float("nan"),
    }

    logger.info(
        "ECG morphology: R-peak rate=%.2f  RR=%.1f±%.1f ms  SNR=%.1f dB",
        results["r_peak_detection_rate"],
        results["mean_rr_interval_ms"],
        results["std_rr_interval_ms"],
        results["mean_snr_db"],
    )
    return results


# ---------------------------------------------------------------------------
# Clinical notes quality metrics
# ---------------------------------------------------------------------------

def text_quality_metrics(
    generated: List[str],
    reference: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Evaluate quality of generated clinical notes.

    Parameters
    ----------
    generated  : list of synthetic clinical notes
    reference  : optional list of real reference notes for BLEU/ROUGE

    Returns
    -------
    dict with:
        ``"avg_length_tokens"`` : mean number of whitespace-split tokens
        ``"vocabulary_size"``   : unique tokens across all generated notes
        ``"bleu_2"``            : corpus BLEU-2 (if reference provided)
        ``"rouge_l"``           : mean ROUGE-L F1 (if reference provided)
    """
    avg_len = np.mean([len(t.split()) for t in generated])
    all_tokens = " ".join(generated).split()
    vocab = len(set(all_tokens))

    results: Dict[str, float] = {
        "avg_length_tokens": float(avg_len),
        "vocabulary_size": float(vocab),
    }

    if reference is not None and len(reference) > 0:
        bleu = _bleu_2(generated, reference)
        rouge = _rouge_l(generated, reference)
        results["bleu_2"] = bleu
        results["rouge_l"] = rouge

    logger.info(
        "Text quality: avg_len=%.1f  vocab=%d  BLEU-2=%.4f  ROUGE-L=%.4f",
        avg_len, vocab,
        results.get("bleu_2", float("nan")),
        results.get("rouge_l", float("nan")),
    )
    return results


def _bleu_2(hypotheses: List[str], references: List[str]) -> float:
    """Corpus-level BLEU-2 (bigram precision with brevity penalty)."""
    from collections import Counter
    import math

    def ngrams(text: str, n: int):
        tokens = text.lower().split()
        return Counter(zip(*[tokens[i:] for i in range(n)]))

    clip_count, total_count = 0, 0
    for hyp in hypotheses[:len(references)]:
        hyp_ngrams = ngrams(hyp, 2)
        # Take best reference match
        ref_ngrams = ngrams(references[0], 2)  # simplified: vs first ref
        for gram, count in hyp_ngrams.items():
            clip_count += min(count, ref_ngrams.get(gram, 0))
            total_count += count

    precision = clip_count / (total_count + 1e-8)
    # Brevity penalty (simplified)
    bp = min(1.0, math.exp(1 - len(references[0].split()) / (np.mean([len(h.split()) for h in hypotheses]) + 1e-8)))
    return float(bp * precision)


def _rouge_l(hypotheses: List[str], references: List[str]) -> float:
    """Mean ROUGE-L F1 using LCS."""
    scores = []
    for hyp, ref in zip(hypotheses, references):
        hyp_tok = hyp.lower().split()
        ref_tok = ref.lower().split()
        lcs = _lcs_length(hyp_tok, ref_tok)
        precision = lcs / (len(hyp_tok) + 1e-8)
        recall = lcs / (len(ref_tok) + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        scores.append(f1)
    return float(np.mean(scores)) if scores else float("nan")


def _lcs_length(a: list, b: list) -> int:
    """Length of longest common subsequence (DP)."""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    # Space-efficient O(min(m,n)) DP
    if m < n:
        a, b, m, n = b, a, n, m
    prev = [0] * (n + 1)
    for ai in a:
        curr = [0] * (n + 1)
        for j, bj in enumerate(b, 1):
            curr[j] = prev[j - 1] + 1 if ai == bj else max(curr[j - 1], prev[j])
        prev = curr
    return prev[n]
