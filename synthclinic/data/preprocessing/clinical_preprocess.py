"""
synthclinic.data.preprocessing.clinical_preprocess
-----------------------------------------------
Preprocessing for clinical note text fed into GPT-2 fine-tuning.

Key steps:
  1. Clean and normalise raw transcription text.
  2. Optionally prepend a specialty prompt token for conditional generation.
  3. Tokenise with the GPT-2 tokeniser.
  4. Wrap in a HuggingFace-compatible Dataset for Trainer API.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def clean_note(text: str) -> str:
    """
    Lightweight clinical text cleaning.

    - Strip leading/trailing whitespace.
    - Collapse multiple blank lines to a single newline.
    - Remove HTML-like tags (sometimes present in transcription exports).
    - Remove non-printable / control characters.
    """
    text = text.strip()
    text = re.sub(r"<[^>]+>", " ", text)               # strip HTML tags
    text = re.sub(r"[^\x20-\x7E\n]", " ", text)        # non-ASCII control chars
    text = re.sub(r"\n{3,}", "\n\n", text)              # collapse blank lines
    text = re.sub(r"[ \t]{2,}", " ", text)              # collapse spaces
    return text


def format_with_prompt(text: str, specialty: Optional[str] = None) -> str:
    """
    Optionally wrap a clinical note with a specialty prefix token.

    This enables *conditional* text generation: at inference time, you can
    prompt the model with "<CARDIOLOGY>" to sample cardiology-specific notes.

    Example
    -------
    >>> format_with_prompt("Patient is a ...", specialty="Cardiology")
    "<CARDIOLOGY> Patient is a ..."
    """
    if specialty and specialty.strip().lower() not in ("", "unknown", "nan"):
        tag = f"<{specialty.upper().replace(' ', '_')}>"
        return f"{tag} {text}"
    return text


# ---------------------------------------------------------------------------
# HuggingFace Dataset wrapper
# ---------------------------------------------------------------------------

class ClinicalNotesDataset:
    """
    Wraps cleaned clinical notes as a HuggingFace ``datasets.Dataset``.

    Parameters
    ----------
    texts:
        Raw transcription strings.
    tokenizer:
        A pre-loaded HuggingFace tokenizer (e.g. ``GPT2Tokenizer``).
    max_length:
        Maximum token length.  Sequences longer than this are truncated.
        Recommended: 512 for GPT-2 (max 1024).
    use_specialty_prefix:
        If ``True``, prepend specialty tokens and pass ``specialties``.
    specialties:
        Parallel list of specialty strings matching ``texts``.
    """

    def __init__(
        self,
        texts: List[str],
        tokenizer,
        max_length: int = 512,
        use_specialty_prefix: bool = False,
        specialties: Optional[List[str]] = None,
    ) -> None:
        self.max_length = max_length
        self.tokenizer = tokenizer

        # Clean + optionally prefix
        cleaned = []
        for i, text in enumerate(texts):
            t = clean_note(text)
            if use_specialty_prefix and specialties is not None:
                t = format_with_prompt(t, specialties[i])
            cleaned.append(t)

        self._texts = cleaned
        logger.info("ClinicalNotesDataset: %d notes prepared", len(self._texts))

    def to_hf_dataset(self):
        """
        Tokenise all texts and return a HuggingFace ``datasets.Dataset``.

        The returned dataset has columns ``input_ids``, ``attention_mask``,
        and ``labels`` (= ``input_ids``, for causal LM training).
        """
        from datasets import Dataset  # type: ignore

        encodings = self.tokenizer(
            self._texts,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors=None,  # return plain Python lists
        )
        # For causal LM, labels == input_ids; mask padding with -100
        labels = [
            [
                token_id if token_id != self.tokenizer.pad_token_id else -100
                for token_id in seq
            ]
            for seq in encodings["input_ids"]
        ]
        encodings["labels"] = labels
        return Dataset.from_dict(encodings)

    def __len__(self) -> int:
        return len(self._texts)


# ---------------------------------------------------------------------------
# Convenience builder
# ---------------------------------------------------------------------------

def build_dataset_from_df(
    df: pd.DataFrame,
    tokenizer,
    max_length: int = 512,
    use_specialty_prefix: bool = True,
) -> "Dataset":  # noqa: F821
    """
    One-liner to go from a cleaned MTSamples DataFrame to a HF Dataset.

    Parameters
    ----------
    df:
        DataFrame with columns ``transcription`` and ``medical_specialty``.
    tokenizer:
        HuggingFace tokenizer instance.
    """
    texts = df["transcription"].tolist()
    specialties = (
        df["medical_specialty"].tolist()
        if "medical_specialty" in df.columns
        else None
    )
    wrapper = ClinicalNotesDataset(
        texts=texts,
        tokenizer=tokenizer,
        max_length=max_length,
        use_specialty_prefix=use_specialty_prefix,
        specialties=specialties,
    )
    return wrapper.to_hf_dataset()
