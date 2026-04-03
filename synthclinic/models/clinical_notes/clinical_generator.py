"""
synthclinic.models.clinical_notes.clinical_generator
-------------------------------------------------
Synthetic clinical note generator via GPT-2 fine-tuning.

Two modes
---------
``use_lora=False`` (default for compute-constrained setups)
    Full fine-tuning of the GPT-2 weights via HuggingFace Trainer API.
    Recommended when training on a small GPU or CPU with < 4 000 notes.

``use_lora=True``
    Parameter-efficient fine-tuning using LoRA (Hu et al., ICLR 2022).
    Adds rank-r update matrices to attention Q/V projections only.
    ~1% of parameters vs full fine-tuning — faster + less memory.

Conditional generation
----------------------
If the dataset was built with ``use_specialty_prefix=True``, the model learns
to generate specialty-conditioned notes.  At inference, pass a ``prompt`` such
as ``"<CARDIOLOGY>"`` to ``generate()`` to steer the output.

Reference
---------
Radford, A. et al. (2019). "Language Models are Unsupervised Multitask Learners."
  OpenAI Blog.  https://openai.com/blog/better-language-models
Hu, E.J. et al. (2022). "LoRA: Low-Rank Adaptation of Large Language Models."
  ICLR 2022. https://arxiv.org/abs/2106.09685
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from synthclinic.base import BaseGenerator

logger = logging.getLogger(__name__)


class ClinicalNotesGenerator(BaseGenerator):
    """
    Generative model for clinical free-text notes (GPT-2 + optional LoRA).

    Parameters
    ----------
    config : dict
        Supported keys:

        ``model_name``         : HuggingFace model ID (default ``"gpt2"``)
        ``use_lora``           : Enable LoRA PEFT (default ``False``)
        ``lora_r``             : LoRA rank (default 8)
        ``lora_alpha``         : LoRA alpha scaling (default 32)
        ``lora_dropout``       : LoRA dropout (default 0.05)
        ``max_length``         : Max token length (default 512)
        ``num_train_epochs``   : Fine-tuning epochs (default 3)
        ``per_device_batch``   : Batch size per device (default 4)
        ``gradient_accumulation_steps``: Accumulate gradients (default 8)
        ``learning_rate``      : Adam lr (default 5e-5)
        ``warmup_steps``       : LR warmup steps (default 100)
        ``output_dir``         : Trainer checkpoint directory (default ``"checkpoints/clinical"``)
        ``use_specialty_prefix``: Include specialty tokens in training (default ``True``)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        config = config or {}
        super().__init__(config)
        self._model = None
        self._tokenizer = None

    # ------------------------------------------------------------------
    # BaseGenerator interface
    # ------------------------------------------------------------------

    def train(self, data) -> Dict[str, float]:
        """
        Fine-tune GPT-2 on clinical notes.

        Parameters
        ----------
        data : pd.DataFrame
            Must contain a ``transcription`` column.
            Optionally a ``medical_specialty`` column for conditional generation.

        Returns
        -------
        dict with ``"train_loss"``
        """
        import pandas as pd
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )

        cfg = self.config
        model_name = cfg.get("model_name", "gpt2")
        max_length = cfg.get("max_length", 512)
        use_specialty = cfg.get("use_specialty_prefix", True)
        output_dir = cfg.get("output_dir", "checkpoints/clinical")
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # ── Tokenizer ──────────────────────────────────────────────────
        logger.info("Loading tokenizer: %s", model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token  # GPT-2 has no pad token

        # Add specialty tokens if needed
        if use_specialty and "medical_specialty" in data.columns:
            specialties = data["medical_specialty"].dropna().unique().tolist()
            special_tokens = [
                f"<{s.upper().replace(' ', '_')}>" for s in specialties
            ]
            tokenizer.add_tokens(special_tokens)
            logger.info("Added %d specialty tokens", len(special_tokens))

        # ── Model ──────────────────────────────────────────────────────
        logger.info("Loading model: %s", model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.resize_token_embeddings(len(tokenizer))

        # ── Optional LoRA ──────────────────────────────────────────────
        if cfg.get("use_lora", False):
            model = self._apply_lora(model, cfg)

        model = model.to(self.device)

        # ── Dataset ────────────────────────────────────────────────────
        from synthclinic.data.preprocessing.clinical_preprocess import build_dataset_from_df

        hf_dataset = build_dataset_from_df(
            df=data,
            tokenizer=tokenizer,
            max_length=max_length,
            use_specialty_prefix=use_specialty,
        )
        # 90/10 train/val split
        split = hf_dataset.train_test_split(test_size=0.1, seed=42)

        # ── Training arguments ─────────────────────────────────────────
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=cfg.get("num_train_epochs", 3),
            per_device_train_batch_size=cfg.get("per_device_batch", 4),
            per_device_eval_batch_size=cfg.get("per_device_batch", 4),
            gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 8),
            learning_rate=cfg.get("learning_rate", 5e-5),
            warmup_steps=cfg.get("warmup_steps", 100),
            weight_decay=0.01,
            logging_steps=50,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            fp16=torch.cuda.is_available(),
            report_to="none",
        )

        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=split["train"],
            eval_dataset=split["test"],
            data_collator=collator,
        )

        logger.info("Fine-tuning %s on %d notes …", model_name, len(split["train"]))
        result = trainer.train()

        self._model = model
        self._tokenizer = tokenizer
        self.is_trained = True

        train_loss = result.training_loss
        logger.info("ClinicalNotesGenerator training complete. Loss=%.4f", train_loss)
        return {"train_loss": train_loss}

    def generate(
        self,
        n_samples: int,
        prompt: Optional[str] = None,
        max_new_tokens: int = 300,
        temperature: float = 0.9,
        top_p: float = 0.92,
        top_k: int = 50,
        **kwargs,
    ) -> List[str]:
        """
        Generate *n_samples* synthetic clinical notes.

        Parameters
        ----------
        n_samples:
            Number of notes to generate.
        prompt:
            Optional conditioning prefix, e.g. ``"<CARDIOLOGY>"``.
            If ``None``, uses ``tokenizer.bos_token``.
        max_new_tokens:
            Maximum tokens to generate per note.
        temperature / top_p / top_k:
            Standard nucleus sampling hyperparameters.

        Returns
        -------
        List[str] of length *n_samples*
        """
        self._require_trained()
        self._model.eval()

        if prompt is None:
            prompt = self._tokenizer.bos_token or ""

        notes = []
        for _ in range(n_samples):
            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=50,
            ).to(self.device)

            with torch.no_grad():
                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    pad_token_id=self._tokenizer.eos_token_id,
                    eos_token_id=self._tokenizer.eos_token_id,
                )

            text = self._tokenizer.decode(
                output_ids[0], skip_special_tokens=False
            )
            # Strip the prompt from the output
            if prompt in text:
                text = text[text.index(prompt) + len(prompt):].strip()
            notes.append(text)

        self._model.train()
        return notes

    def save(self, path: str | Path) -> None:
        self._require_trained()
        out_dir = Path(path) / "clinical_generator"
        out_dir.mkdir(parents=True, exist_ok=True)

        self._model.save_pretrained(out_dir / "model")
        self._tokenizer.save_pretrained(out_dir / "tokenizer")

        with open(out_dir / "config.json", "w") as f:
            json.dump(self.config, f, indent=2)

        logger.info("ClinicalNotesGenerator saved to %s", out_dir)

    def load(self, path: str | Path) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        in_dir = Path(path) / "clinical_generator"
        if not in_dir.exists():
            in_dir = Path(path)

        self._tokenizer = AutoTokenizer.from_pretrained(in_dir / "tokenizer")
        self._model = AutoModelForCausalLM.from_pretrained(
            in_dir / "model"
        ).to(self.device)
        self._model.eval()
        self.is_trained = True
        logger.info("ClinicalNotesGenerator loaded from %s", in_dir)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_lora(model, cfg: Dict[str, Any]):
        """Apply LoRA adapters to GPT-2 attention layers."""
        from peft import LoraConfig, TaskType, get_peft_model  # type: ignore

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=cfg.get("lora_r", 8),
            lora_alpha=cfg.get("lora_alpha", 32),
            lora_dropout=cfg.get("lora_dropout", 0.05),
            target_modules=["c_attn"],  # GPT-2 attention projection
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        return model
