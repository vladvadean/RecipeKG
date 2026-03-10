# Recipe Analyzer — Setup & Run Guide

## Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended) or CPU

---

## 1. Install Dependencies

```bash
python -m venv venv
```

```bash
pip install -r requirements.txt
```

---

## 2. Place Required Model & Data Files

Ensure the following files are in the **same directory** as your scripts:

| File                                          | Description                                    |
| --------------------------------------------- | ---------------------------------------------- |
| `KG.graphml`                                  | Knowledge graph                                |
| `prepared_recipes.csv`                        | Recipe dataset                                 |
| `food.csv`                                    | Food categories (requires a `Category` column) |
| `node_type_vocab_mean.json`                   | Node type vocabulary                           |
| `edge_relationship_vocab_mean.json`           | Edge relationship vocabulary                   |
| `subgraph_embeddings_mean.json`               | Pre-computed subgraph embeddings               |
| `gcn_model_mean.pth`                          | GCN model weights                              |
| `t5_joint_settings_2023_05_15-09_02_53_AM.pt` | T5 classifier weights                          |
| `index.html`                                  | Frontend UI                                    |

---

## 3. Project Structure

```
project/
├── app.py          # RecipeProcessor + model definitions
├── main.py         # FastAPI server
├── index.html      # Frontend UI
├── KG.graphml
├── food.csv
├── prepared_recipes.csv
├── node_type_vocab_mean.json
├── edge_relationship_vocab_mean.json
├── subgraph_embeddings_mean.json
├── gcn_model_mean.pth
└── t5_joint_settings_2023_05_15-09_02_53_AM.pt
```

---

## 4. Run the Server

```bash
uvicorn main:app --reload
```

All models load on startup — this may take **1–3 minutes**. Watch the console for:

```
All models loaded successfully!
```

---

## 5. Use the App

- **Frontend UI:** Open [http://localhost:8000](http://localhost:8000) in your browser
- **API directly:** Send a `POST` request to `http://localhost:8000/echo` with the body:

```json
{ "message": "your recipe instructions here" }
```
