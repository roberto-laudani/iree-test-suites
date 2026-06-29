#!/usr/bin/env python3
"""Regenerates the Whisper-tiny.en IREE quality-test artifacts.

Produces in --out-dir:
  model.mlir              exported torch-dialect MLIR
  real_weights.irpa       externalized parameters
  inference_input.0a.bin  f32 log-mel features, shape [1, 80, 3000]
  inference_input.0b.bin  int64 decoder_input_ids, shape [1, dec_len]
  inference_output.0.bin  f32 logits, shape [1, dec_len, 51864]
"""

import argparse
import sys
from pathlib import Path

import torch

from datasets import Audio, load_dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import iree.turbine.aot as aot

MODEL_ID = "openai/whisper-tiny.en"
MODEL_REVISION = "87c7102498dcde7456f24cfd30239ca606ed9063"

DATASET_ID = "hf-internal-testing/librispeech_asr_dummy"
DATASET_REVISION = "5be91486e11a2d616f4ec5db8d3fd248585ac07a"
DATASET_CONFIG = "clean"
DATASET_SPLIT = "validation"
SAMPLING_RATE = 16000
SAMPLE_INDEX = 0


def create_causal_mask_4d(seq_len, dtype=torch.float32):
    """Precomputed 4D additive causal mask, shape [1, 1, seq_len, seq_len]."""
    mask_fill = torch.finfo(dtype).min
    mask = torch.full((seq_len, seq_len), mask_fill, dtype=dtype)
    return torch.triu(mask, diagonal=1)[None, None]


def load_reference_audio():
    """Load the pinned LibriSpeech dummy sample as a 16 kHz mono waveform."""
    import io

    import soundfile as sf

    ds = load_dataset(
        DATASET_ID, DATASET_CONFIG, split=DATASET_SPLIT, revision=DATASET_REVISION
    )
    row = ds.cast_column("audio", Audio(decode=False))[SAMPLE_INDEX]
    audio_field = row["audio"]
    src = (
        io.BytesIO(audio_field["bytes"])
        if audio_field.get("bytes")
        else audio_field["path"]
    )
    audio, sampling_rate = sf.read(src, dtype="float32", always_2d=False)
    assert (
        sampling_rate == SAMPLING_RATE
    ), f"expected {SAMPLING_RATE} Hz, got {sampling_rate}"
    return audio, row["text"]


class WhisperLogits(torch.nn.Module):
    """Single teacher-forced forward producing logits."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_features, decoder_input_ids):
        encoder_hidden_states = self.model.model.encoder(
            input_features
        ).last_hidden_state
        mask = create_causal_mask_4d(
            decoder_input_ids.shape[1], dtype=encoder_hidden_states.dtype
        )
        sequence_output = self.model.model.decoder(
            input_ids=decoder_input_ids,
            attention_mask=mask,
            encoder_hidden_states=encoder_hidden_states,
            use_cache=False,
        ).last_hidden_state
        return self.model.proj_out(sequence_output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).parent / "artifacts",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model {MODEL_ID}@{MODEL_REVISION}", file=sys.stderr)
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype=torch.float32,
        attn_implementation="eager",
    )
    model.eval()
    processor = WhisperProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)

    print(
        f"Loading reference audio {DATASET_ID}@{DATASET_REVISION}[{SAMPLE_INDEX}]",
        file=sys.stderr,
    )
    audio, transcript = load_reference_audio()
    input_features = processor(
        audio, sampling_rate=SAMPLING_RATE, return_tensors="pt"
    ).input_features.to(torch.float32)
    print(f"  transcript     {transcript!r}", file=sys.stderr)
    print(
        f"  input features {tuple(input_features.shape)} {input_features.dtype}",
        file=sys.stderr,
    )

    # Teacher-forced decoder input: forced prefix and ground-truth transcript.
    tokenizer = processor.tokenizer
    prefix = [
        model.config.decoder_start_token_id,
        tokenizer.convert_tokens_to_ids("<|notimestamps|>"),
    ]
    transcript_ids = tokenizer(transcript, add_special_tokens=False).input_ids
    decoder_input_ids = torch.tensor([prefix + transcript_ids], dtype=torch.int64)
    print(
        f"  decoder ids    {tuple(decoder_input_ids.shape)} {decoder_input_ids.dtype}",
        file=sys.stderr,
    )

    print("Running eager reference forward", file=sys.stderr)
    wrapper = WhisperLogits(model)
    with torch.no_grad():
        ref_logits = wrapper(input_features, decoder_input_ids)
    ref_logits = ref_logits.to(torch.float32).contiguous()
    print(
        f"  output logits  {tuple(ref_logits.shape)} {ref_logits.dtype}",
        file=sys.stderr,
    )

    # Dump raw reference input/output data.
    print("Writing reference input/output .bin files", file=sys.stderr)
    (args.out_dir / "inference_input.0a.bin").write_bytes(
        input_features.numpy().astype("<f4").tobytes()
    )
    (args.out_dir / "inference_input.0b.bin").write_bytes(
        decoder_input_ids.numpy().astype("<i8").tobytes()
    )
    (args.out_dir / "inference_output.0.bin").write_bytes(
        ref_logits.numpy().astype("<f4").tobytes()
    )

    # Export to MLIR with weights externalized into a separate .irpa.
    print("Exporting to MLIR", file=sys.stderr)
    aot.externalize_module_parameters(wrapper)
    export_output = aot.export(wrapper, args=(input_features, decoder_input_ids))
    export_output.save_mlir(str(args.out_dir / "model.mlir"))
    aot.save_module_parameters(str(args.out_dir / "real_weights.irpa"), wrapper)

    print(f"Artifacts successfully written to {args.out_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
