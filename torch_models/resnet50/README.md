# ResNet-50 regression test

End-to-end regression for
[`microsoft/resnet-50`](https://huggingface.co/microsoft/resnet-50). IREE compiles
the committed MLIR, runs it on CPU, and compares the output logits against
reference values fetched from the Hugging Face Hub.

## Files

| File | Description |
| --- | --- |
| `resnet50_quality_cpu.json` | Quality test definition |
| `modules/resnet50_cpu.json` | Module definition and compiler flags |
| `model.mlir` | Exported torch-dialect program, parameters externalized |
| `generate.py` | Reproducer for the artifacts above and the parameters |
| `requirements.txt` | Pinned toolchain for `generate.py` |

Model parameters and reference inputs/outputs are hosted on the Hugging Face Hub
at [`roofline/iree-regression-models`](https://huggingface.co/roofline/iree-regression-models),
pinned by revision in `resnet50_quality_cpu.json` and fetched at test time:

| File | Description |
| --- | --- |
| `resnet50/real_weights.irpa` | Externalized model parameters |
| `resnet50/inference_input.0.bin` | Preprocessed `huggingface/cats-image` input sample (`1x3x224x224xf32`) |
| `resnet50/inference_output.0.bin` | Expected output logits (`1x1000xf32`) |

## Reproducing artifacts

```bash
pip install -r resnet50/requirements.txt
python3 resnet50/generate.py
```
