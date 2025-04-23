# 🧠 LLM Medicine Substitutes

A full pipeline for fine-tuning a Large Language Model (LLM) using Indian medicines data to recommend substitutes based on composition similarity. This project uses FAISS with TF-IDF vectors to find similar medicines and formats the results for instruction-based LLM tuning.

---

## 📂 Project Structure

```
llm-medicine-substitutes/
├── notebooks/                     # Jupyter notebooks for processing
│   ├── 01_preprocess_dataset.ipynb
│   ├── 02_tfidf_faiss_similarity.ipynb
│   └── 03_generate_jsonl.ipynb
├── outputs/ (compressed)                      # Processed data and outputs
│   ├── medicines_cleaned.csv
│   ├── medicine_comp_substitue.csv
│   └── instructions.jsonl
├── requirements.txt              # Python dependencies
└── README.md
```

---

## 🚀 Pipeline Overview

### 1. Preprocess Dataset
- Merge and clean composition fields.
- Dataset available at [Kaggle Indian Medicine Dataset](https://www.kaggle.com/datasets/riturajsingh2004/extensive-a-z-medicines-dataset-of-india).
- Normalize drug names and dosage formats.

### 2. TF-IDF Vectorization + FAISS Similarity
- Compute cosine similarity using FAISS (CPU).
- Extract top-5 substitutes per drug.

### 3. Instruction Format for LLM
- Converts drug + composition into `"instruction"` prompts.
- Produces Hugging Face-compatible `.jsonl` for SFT fine-tuning.

---

## 🔄 Example Instruction Format

```json
{
  "instruction": "Suggest alternative medicines for 'Augmentin 625 Duo Tablet', which contains the composition: amoxycillin clavulanic acid.",
  "input": "",
  "output": "Here are some similar alternatives: Augmentin 1000 Duo Tablet, Amoxyclav 625 Tablet, Azithral 500 Tablet, ..."
}
```

---

## 🧰 Dependencies

Install everything using:
```bash
pip install -r requirements.txt
```

---

## 🏁 How to Run

1. Run each notebook in order (`01_ → 03_`)
2. Fine-tune any model (like DeepSeek, Mistral, LLaMA) using `outputs/instructions.jsonl`

---

## 🤝 Acknowledgements

- [Kaggle Indian Medicine Dataset](https://www.kaggle.com/datasets/riturajsingh2004/extensive-a-z-medicines-dataset-of-india)
- [FAISS (Facebook AI Similarity Search)](https://github.com/facebookresearch/faiss)
- Hugging Face 🤗 ecosystem for tokenizer, datasets, and training

---

## 📜 License

MIT License (you may modify/use for non-commercial or academic purposes)