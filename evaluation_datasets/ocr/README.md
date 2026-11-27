# OCR benchmark dataset

Create `manifest.jsonl` beside this file. Every line represents one verified
page or image and must include:

```json
{"id":"arabic-scan-1","pdf":"samples/arabic_scan.pdf","page_number":1,"language":"ara","kind":"scanned_pdf","ground_truth":"النص الصحيح كما يظهر في الصفحة","expected_tables":0}
```

Use `image` instead of `pdf` for a standalone image. Ground truth must be
human-verified; using another OCR engine's output as truth invalidates CER/WER.
Include at least one sample for each required slice:

- Arabic scanned text
- English scanned text
- Arabic-English mixed text
- a low-quality or skewed scan
- a table page

Run the local benchmark without any OpenRouter calls:

```bash
uv run python scripts/benchmark_ocr.py \
  --manifest evaluation_datasets/ocr/manifest.jsonl \
  --providers paddle tesseract surya
```

The script writes detailed JSON/CSV plus a provider summary under
`data/ocr_benchmark/`. Lower CER and WER are better. Compare latency, detected
boxes, table output, and failed samples before choosing the default provider.
