# Qwen3-0.6B regression test

End-to-end regression for
[`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B). IREE compiles the
committed MLIR, runs a single static-shape prefill forward on CPU, and compares
the output logits against reference values fetched from the Hugging Face Hub.

## Files

| File | Description |
| --- | --- |
| `qwen3-600m_quality_cpu.json` | Quality test definition |
| `modules/qwen3-600m_cpu.json` | Module definition and compiler flags |
| `model.mlir` | Exported torch-dialect program, parameters externalized |

Model parameters and reference inputs/outputs are hosted on the Hugging Face Hub
at [`roofline/iree-regression-models`](https://huggingface.co/roofline/iree-regression-models),
pinned by revision in `qwen3-600m_quality_cpu.json` and fetched at test time:

| File | Description |
| --- | --- |
| `qwen3-600m/real_weights.irpa` | Externalized model parameters |
| `qwen3-600m/inference_input.0.bin` | Tokenized prompt `"The capital of France is"` (`1x5xi64`) |
| `qwen3-600m/inference_output.0.bin` | Expected output logits (`1x5x151936xf32`) |

## Reproducing artifacts

```bash
pip install -r qwen3-600m/requirements.txt
python3 qwen3-600m/generate.py
```
