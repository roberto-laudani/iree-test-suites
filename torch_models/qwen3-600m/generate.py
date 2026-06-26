#!/usr/bin/env python3
"""Regenerates the Qwen3-0.6B IREE quality-test artifacts.

Produces in --out-dir:
  model.mlir              exported torch-dialect MLIR
  real_weights.irpa       externalized parameters
  inference_input.0.bin   int64 input_ids, shape [1, seq_len]
  inference_output.0.bin  f32 logits, shape [1, seq_len, 151936]
"""

import argparse
import sys
from pathlib import Path

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
import iree.turbine.aot as aot

MODEL_ID = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
PROMPT = "The capital of France is"


def create_causal_mask_4d(seq_len, dtype=torch.float32):
    """Precomputed 4D additive causal mask, shape [1, 1, seq_len, seq_len]."""
    mask_fill = torch.finfo(dtype).min
    mask = torch.full((seq_len, seq_len), mask_fill, dtype=dtype)
    return torch.triu(mask, diagonal=1)[None, None]


class CausalLM(torch.nn.Module):
    """Single forward producing logits."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        return self.model(
            input_ids=input_ids,
            attention_mask=create_causal_mask_4d(input_ids.shape[1]),
            use_cache=False,
        ).logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).parent / "artifacts",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model {MODEL_ID}@{MODEL_REVISION} (f32)", file=sys.stderr)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype=torch.float32,
        attn_implementation="eager",
    )
    model.eval()

    # Tokenize the prompt with the checkpoint's own tokenizer.
    print(f"Tokenizing prompt {PROMPT!r}", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(torch.int64)
    print(
        f"  input  shape  {tuple(input_ids.shape)} {input_ids.dtype}", file=sys.stderr
    )
    print(f"  input  ids    {input_ids[0].tolist()}", file=sys.stderr)
    print(
        f"  input  tokens {tokenizer.convert_ids_to_tokens(input_ids[0].tolist())}",
        file=sys.stderr,
    )

    print("Running eager reference forward", file=sys.stderr)
    with torch.no_grad():
        ref_logits = model(
            input_ids=input_ids,
            attention_mask=create_causal_mask_4d(input_ids.shape[1]),
            use_cache=False,
        ).logits
    ref_logits = ref_logits.to(torch.float32).contiguous()
    print(
        f"  output shape  {tuple(ref_logits.shape)} {ref_logits.dtype}", file=sys.stderr
    )
    # Greedy argmax per position; the last position is the model's next-token.
    pred_ids = ref_logits[0].argmax(dim=-1).tolist()
    print(f"  output ids    {pred_ids}", file=sys.stderr)
    print(
        f"  output tokens {tokenizer.convert_ids_to_tokens(pred_ids)}", file=sys.stderr
    )
    print(f"  next token    {tokenizer.decode([pred_ids[-1]])!r}", file=sys.stderr)

    # Dump raw reference input/output data.
    print("Writing reference input/output .bin files", file=sys.stderr)
    (args.out_dir / "inference_input.0.bin").write_bytes(
        input_ids.numpy().astype("<i8").tobytes()
    )
    (args.out_dir / "inference_output.0.bin").write_bytes(
        ref_logits.numpy().astype("<f4").tobytes()
    )

    # Export to MLIR with weights externalized into a separate .irpa.
    print("Exporting to MLIR", file=sys.stderr)
    wrapper = CausalLM(model)
    aot.externalize_module_parameters(wrapper)
    export_output = aot.export(wrapper, args=(input_ids,))
    export_output.save_mlir(str(args.out_dir / "model.mlir"))
    aot.save_module_parameters(str(args.out_dir / "real_weights.irpa"), wrapper)

    print(f"Artifacts successfully written to {args.out_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
