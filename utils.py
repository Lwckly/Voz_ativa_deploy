#!/usr/bin/env python3
"""
train_lora_whisper.py

Usage example:
python train_lora_whisper.py \
  --train_json dataset/train.jsonl \
  --valid_json dataset/valid.jsonl \
  --output_dir ./whisper_lora_out \
  --per_device_train_batch_size 4 \
  --num_train_epochs 3 \
  --adapter_init_path /mnt/data/whisper_lora_base2_extracted/whisper_lora_base

Input dataset JSONL format (one JSON per line):
{"audio_filepath": "/abs/path/to/wav1.wav", "duration": 12.3, "text": "transcription text..."}
"""

import argparse
import os
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import numpy as np
from datasets import load_dataset, Audio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
import evaluate

# -------------------------
# Default adapter path (from your uploaded zip)
DEFAULT_ADAPTER_INIT = "/mnt/data/whisper_lora_base2_extracted/whisper_lora_base"
# -------------------------

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # features: list of dicts containing 'input_features' and 'labels'
        input_features = [f["input_features"][0] if isinstance(f["input_features"], list) else f["input_features"] for f in features]
        batch = self.processor.feature_extractor.pad({"input_features": input_features}, return_tensors="pt")
        labels = [f["labels"] for f in features]
        # pad labels
        labels_batch = self.processor.tokenizer.pad({"input_ids": labels}, return_tensors="pt", padding=True)
        # rename to labels and replace padding token id with -100
        labels_ids = labels_batch["input_ids"].clone()
        labels_ids[labels_ids == self.processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels_ids
        return batch

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_json", type=str, required=True, help="train jsonl/csv (audio_filepath,text)")
    p.add_argument("--valid_json", type=str, required=False, default=None, help="valid jsonl/csv (optional)")
    p.add_argument("--base_model", type=str, default="openai/whisper-base")
    p.add_argument("--adapter_init_path", type=str, default=DEFAULT_ADAPTER_INIT,
                   help="existing adapter folder to resume from (optional). If empty string -> fresh LoRA.")
    p.add_argument("--output_dir", type=str, default="./whisper_lora_out")
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--per_device_eval_batch_size", type=int, default=4)
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--save_total_limit", type=int, default=3)
    p.add_argument("--logging_steps", type=int, default=100)
    p.add_argument("--r", type=int, default=8, help="LoRA rank")
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--target_modules", nargs="+", default=["q_proj","k_proj","v_proj","o_proj"])
    p.add_argument("--do_merge", action="store_true", help="merge LoRA into base and save a full model at end")
    return p.parse_args()

def prepare_datasets(train_json, valid_json=None, sampling_rate=16000):
    # load json/csv into datasets
    data_files = {"train": train_json}
    if valid_json:
        data_files["validation"] = valid_json
    ds = load_dataset("json", data_files=data_files) if train_json.lower().endswith(".jsonl") or train_json.lower().endswith(".json") else load_dataset("csv", data_files=data_files)
    # cast audio column to Audio with desired sampling_rate
    ds = ds.cast_column("audio", Audio(sampling_rate=sampling_rate)) if "audio" in ds["train"].column_names else ds
    # If dataset uses 'audio_filepath' use that column to create audio dict
    # We'll expect "audio_filepath" in the input JSONL; if audio column doesn't exist we'll map it.
    # Ensure each example has 'audio' field as expected by processor
    def ensure_audio(example):
        if "audio" in example:
            return example
        # if 'audio_filepath' exists, datasets Audio will handle if cast, otherwise read raw path using soundfile later
        if "audio_filepath" in example:
            return example
        raise ValueError("Dataset must include either 'audio' (Audio) column or 'audio_filepath' field.")
    ds = ds.map(ensure_audio)
    return ds

def preprocess_function(batch, processor: WhisperProcessor, audio_column_name="audio", text_column_name="text"):
    # batch audio can be path or dict; using processor
    audios = []
    for a in batch[audio_column_name]:
        # if dataset Audio: a is dict with 'array' and 'sampling_rate'
        if isinstance(a, dict) and "array" in a:
            audios.append({"array": a["array"], "sampling_rate": a["sampling_rate"]})
        else:
            # if a is a path string
            audios.append({"path": a})
    input_features = processor.feature_extractor(
        [x["array"] if "array" in x else x["path"] for x in audios],
        sampling_rate=16000,
        return_tensors=None,
    )["input_features"]
    # tokenize labels
    with processor.as_target_processor():
        labels = [processor.tokenizer(t)["input_ids"] for t in batch[text_column_name]]
    # return in the format expected by our collator
    out = {"input_features": [{"input_features": f} for f in input_features], "labels": labels}
    return out

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load processor and base model
    print("Loading processor and base model:", args.base_model)
    processor = WhisperProcessor.from_pretrained(args.base_model, language="portuguese", task="transcribe")

    # Load base model (use float32 to avoid precision issues; you can switch to 8-bit later)
    base = WhisperForConditionalGeneration.from_pretrained(args.base_model)
    base.config.forced_decoder_ids = processor.get_forced_decoder_ids(language="portuguese", task="transcribe")
    base.config.suppress_tokens = processor.get_suppress_tokens()

    # If adapter exists, load PeftModel (resume). Else attach fresh LoRA using get_peft_model
    model = None
    if args.adapter_init_path and os.path.isdir(args.adapter_init_path):
        print("Adapter init path found, resuming from adapter:", args.adapter_init_path)
        model = PeftModel.from_pretrained(base, args.adapter_init_path)
    else:
        print("No adapter found. Preparing a fresh LoRA wrapper.")
        # optional prepare_model_for_kbit_training(base)  # if using 8-bit
        lora_config = LoraConfig(
            r=args.r,
            lora_alpha=args.lora_alpha,
            target_modules=args.target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type="SEQ_2_SEQ_LM",
        )
        model = get_peft_model(base, lora_config)
        print("LoRA wrapper attached. Trainable params:")
        model.print_trainable_parameters()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Prepare datasets
    print("Loading datasets...")
    ds = prepare_datasets(args.train_json, args.valid_json)
    # cast column names
    # We expect either 'audio' column of type Audio or 'audio_filepath' column with paths
    audio_col = "audio" if "audio" in ds["train"].column_names else "audio_filepath"
    text_col = "text" if "text" in ds["train"].column_names else "sentence"

    print("Preprocessing dataset (this may take a while)...")
    # Map preprocess in batched mode
    def map_fn(batch):
        return preprocess_function(batch, processor, audio_column_name=audio_col, text_column_name=text_col)

    tokenized = ds.map(map_fn, batched=True, remove_columns=ds["train"].column_names, num_proc=1)

    # Data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # Metrics
    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        # decode
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        # replace -100 in labels
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    # Training args
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        predict_with_generate=True,
        generation_max_length=448,
        evaluation_strategy="epoch" if args.valid_json else "no",
        save_strategy="epoch",
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        save_total_limit=args.save_total_limit,
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
        push_to_hub=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"] if "validation" in tokenized else None,
        data_collator=data_collator,
        tokenizer=processor.tokenizer,
        compute_metrics=compute_metrics if args.valid_json else None,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save LoRA adapter (recommended backup)
    adapter_out = os.path.join(args.output_dir, "adapter")
    print(f"Saving LoRA adapter to {adapter_out} ...")
    # If model is a PeftModel wrapper, calling save_pretrained will save the adapter
    model.save_pretrained(adapter_out)
    print("Adapter saved.")

    # Optionally merge and save full model
    if args.do_merge:
        print("Merging LoRA into base model (this will produce a full model snapshot)...")
        # ensure peft wrapper exists and has merge method
        try:
            merged = model.merge_and_unload()
            merged_out = os.path.join(args.output_dir, "merged_full")
            merged.save_pretrained(merged_out)
            print("Merged full model saved to", merged_out)
        except Exception as e:
            print("Merge failed:", e)

    print("Training complete. Adapter folder:", adapter_out)
    print("To load later for inference:")
    print("  base = WhisperForConditionalGeneration.from_pretrained('<base>')")
    print("  model = PeftModel.from_pretrained(base, '<adapter_folder_path>')")

if __name__ == "__main__":
    main()
