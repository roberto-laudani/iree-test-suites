#!/usr/bin/env python3
"""Regenerates the ResNet-50 IREE quality-test artifacts.

Produces in --out-dir:
  model.mlir              exported torch-dialect MLIR
  real_weights.irpa       externalized parameters
  inference_input.0.bin   f32 pixel_values, shape [1, 3, 224, 224]
  inference_output.0.bin  f32 logits, shape [1, 1000]
"""

import argparse
import sys
import warnings
from pathlib import Path

import torch

from datasets import load_dataset
from transformers import AutoModelForImageClassification, ConvNextImageProcessorPil
import iree.turbine.aot as aot

MODEL_ID = "microsoft/resnet-50"
MODEL_REVISION = "34c2154c194f829b11125337b98c8f5f9965ff19"
DATASET_ID = "huggingface/cats-image"
DATASET_REVISION = "4613f5f1d3642cc2d56ffdf1b58c9d0f912cdc1f"


class Classifier(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        return self.model(pixel_values=pixel_values).logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).parent / "artifacts",
    )
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--dataset-revision", default=DATASET_REVISION)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model {MODEL_ID}@{args.model_revision} (f32)", file=sys.stderr)
    model = AutoModelForImageClassification.from_pretrained(
        MODEL_ID, revision=args.model_revision, dtype=torch.float32
    )
    model.eval()

    # Preprocess with the checkpoint's own processor to match its documented preprocessing.
    print(
        f"Loading and preprocessing sample image from {args.dataset_id}@{args.dataset_revision}",
        file=sys.stderr,
    )
    image = load_dataset(args.dataset_id, revision=args.dataset_revision)["test"][
        "image"
    ][0].convert("RGB")
    processor = ConvNextImageProcessorPil.from_pretrained(
        MODEL_ID, revision=args.model_revision
    )
    pixel_values = (
        processor(image, return_tensors="pt")["pixel_values"]
        .to(torch.float32)
        .contiguous()
    )

    # Capture eager logits before externalizing params so numerics match the export.
    print("Running eager reference forward", file=sys.stderr)
    with torch.no_grad():
        ref_logits = model(pixel_values=pixel_values).logits
    ref_logits = ref_logits.to(torch.float32).contiguous()

    # Dump raw reference input/output data.
    print("Writing reference input/output .bin files", file=sys.stderr)
    (args.out_dir / "inference_input.0.bin").write_bytes(
        pixel_values.numpy().astype("<f4").tobytes()
    )
    (args.out_dir / "inference_output.0.bin").write_bytes(
        ref_logits.numpy().astype("<f4").tobytes()
    )

    # Export to MLIR with weights externalized into a separate .irpa.
    print("Exporting to MLIR", file=sys.stderr)
    wrapper = Classifier(model)
    aot.externalize_module_parameters(wrapper)
    export_output = aot.export(wrapper, args=(pixel_values,))
    export_output.save_mlir(str(args.out_dir / "model.mlir"))
    aot.save_module_parameters(str(args.out_dir / "real_weights.irpa"), wrapper)

    print(f"Artifacts successfully written to {args.out_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
