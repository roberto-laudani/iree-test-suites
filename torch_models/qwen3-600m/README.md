# Qwen3-0.6B regression test

End-to-end regression for
[`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B). IREE compiles the
committed MLIR, runs a single static-shape prefill forward on CPU, and compares
the output logits against committed reference values.

## Files

| File | Description |
| --- | --- |
| `qwen3-600m_quality_cpu.json` | Quality test definition |
| `modules/qwen3-600m_cpu.json` | Module definition and compiler flags |
| `model.mlir` | Exported torch-dialect program, parameters externalized |
| `inference_input.0.bin` | Tokenized prompt `"The capital of France is"` (`1x5xi64`) |
| `inference_output.0.bin` | Expected output logits (`1x5x151936xf32`) |

Model parameters are hosted on the Hugging Face Hub at
[`roofline/iree-regression-models`](https://huggingface.co/roofline/iree-regression-models)
as `qwen3-600m/real_weights.irpa`, pinned by revision in `qwen3-600m_quality_cpu.json`
and fetched at test time.


## Reproducing artifacts

```bash
pip install -r qwen3-600m/requirements.txt
python3 qwen3-600m/generate.py
```
