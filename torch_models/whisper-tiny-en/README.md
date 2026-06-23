# Whisper-tiny.en regression test

End-to-end regression for
[`openai/whisper-tiny.en`](https://huggingface.co/openai/whisper-tiny.en). IREE
compiles the committed MLIR, runs a single static-shape, teacher-forced forward
on CPU, and compares the output logits against
committed reference values.

## Files

| File | Description |
| --- | --- |
| `whisper-tiny-en_quality_cpu.json` | Quality test definition |
| `modules/whisper-tiny-en_cpu.json` | Module definition and compiler flags |
| `model.mlir` | Exported torch-dialect program |
| `inference_input.0a.bin` | Log-mel features (`1x80x3000xf32`) |
| `inference_input.0b.bin` | Decoder input ids (`1x33xi64`) |
| `inference_output.0.bin` | Expected output logits (`1x33x51864xf32`) |

Model parameters are hosted on the Hugging Face Hub at
[`roofline/iree-regression-models`](https://huggingface.co/roofline/iree-regression-models)
as `whisper-tiny-en/real_weights.irpa`, pinned by revision in
`whisper-tiny-en_quality_cpu.json` and fetched at test time.


## Reproducing artifacts

```bash
pip install -r whisper-tiny-en/requirements.txt
python3 whisper-tiny-en/generate.py
```
