# Conversational Constrained Clustering (C3) Framework

C3 is a framework designed to bridge the gap between human intent and clustering algorithms. Instead of manually specifying numerical parameters or millions of pairwise constraints, the user interacts with an AI agent in a natural language chat. The agent translates user instructions (e.g., *"Separate negative reviews from positive ones"* or *"Group computer science papers together"*) into pairwise Constraints (Must-link/Cannot-link) on a small sampled subset of data. These constraints are then propagated to the entire dataset using **Pairwise Constrained K-Means (PCK-Means)**.

---

## 🚀 Key Features

* **Conversational Control:** Use natural language feedback to direct clustering.
* **Aspect-Based Clustering:** Dynamically transition between cross-cutting dimensions (e.g., Topic vs. Sentiment) on the same dataset.
* **Streamlit Interactive UI:** A premium, fully-featured, web-based chat and interactive 2D visualization dashboard for clustering exploration.
* **Soft Constraint PCK-Means:** Custom native Python implementation of PCK-Means with:
  * Efficient **Constraint Propagation** (via Union-Find Transitive Closure).
  * Robust **K-Means++** initialization.
  * Adaptive constraint penalty weights.
* **Extensibility:** Multi-provider LLM client support (Ollama, OpenAI, Gemini, Anthropic, GitHub Models) and fallback rule-based execution for zero-configuration local runs.
* **Evaluation Sandbox:** Simulation environments simulating human user feedback using an LLM User Simulator.
* **Interactive Widgets Dashboard:** An inline Jupyter Notebook UI built with `ipywidgets`.

---

## 📂 Repository Structure

```
conversational-clustering/
├── src/
│   ├── agent/
│   │   ├── base_agent.py          # Abstract LLM agent class
│   │   ├── local_agent.py         # Ollama LLM agent
│   │   └── cloud_agent.py         # OpenAI, Gemini, Anthropic, GitHub Models agents with JSON parser
│   ├── embeddings/
│   │   ├── base_embeddings.py     # Abstract embeddings class
│   │   └── local_embeddings.py    # SentenceTransformers embeddings with Sklearn fallback
│   ├── clustering/
│   │   ├── base_clustering.py     # Abstract constrained clustering wrapper
│   │   ├── pckmeans.py            # Custom PCK-Means implementation (robust)
│   │   └── external_wrapper.py    # Wrapper for copkmeans with PCK-Means fallback
│   ├── dataset/
│   │   └── loader.py              # CSV/JSON dataset loader supporting multi-aspects
│   ├── evaluation/
│   │   ├── metrics.py             # Evaluation metrics (ARI, NMI, ICU, AUAC)
│   │   └── simulator.py           # LLM User Simulator for automated evaluations
│   ├── pipeline.py                # Orchestrates sampling, agent, constraints, and clustering
│   └── run_experiments.py         # Script to run simulations and compare baselines
├── notebooks/
│   ├── interactive_demo.ipynb     # Interactive Jupyter Notebook dashboard (ipywidgets)
│   └── experiments.ipynb          # Automated benchmark comparison notebook (C3 vs Baselines)
├── data/                          # Directory to place your custom datasets (not tracked in Git)
├── tests/                         # Pytest unit test suite
├── requirements.txt               # Package dependencies
├── app.py                         # Streamlit premium interactive web application
└── README.md                      # Usage instructions
```

---

## 🛠️ Installation & Setup

1. **Clone the Repository & Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure API Keys (Optional):**
   To run C3 with commercial or GitHub model APIs, set the respective environment variables:
   ```bash
   # Windows PowerShell
   $env:OPENAI_API_KEY="your-openai-key"
   $env:GEMINI_API_KEY="your-gemini-key"
   $env:ANTHROPIC_API_KEY="your-anthropic-key"
   $env:GITHUB_TOKEN="your-github-token"
   ```
   *Note: If no API keys are configured, you can use a local Ollama server (e.g., Qwen2.5) for both Q&A and Clustering agents in the Streamlit application.*

---

## 💾 Dataset Loading & Preprocessing

> [!IMPORTANT]
> **No datasets are included in this Git repository.** You must supply your own custom dataset file (CSV, JSON, or JSONL) to test and run the application. Place your data file under the `data/` folder or specify its path directly in the configurations.

The C3 framework natively supports loading datasets in **CSV**, **JSON**, and **JSONL** formats via the unified loader interface in [`src/dataset/loader.py`](file:///c:/LocalRepo/conversational-clustering/src/dataset/loader.py).

### 1. Data Schema Requirements
To load custom data, format your file as follows:
* **Text Column/Field:** The loader expects a field named precisely `text` or `content` to serve as the document content.
* **Aspect/Ground-Truth Columns:** Any other columns or keys in the data file are automatically treated as ground-truth **Aspects** (e.g., `category`, `sentiment`, `methodology`) and loaded as target labels for computing evaluation metrics (ARI/NMI/ACC).

#### Examples:
* **CSV Format:**
  ```csv
  text,category,sentiment
  "Great battery life!","battery","positive"
  "Muffled sound quality.","audio","negative"
  ```
* **JSON Format (List of Objects):**
  ```json
  [
    {"text": "Sample text A", "category": "Math"},
    {"text": "Sample text B", "category": "Physics"}
  ]
  ```
* **JSONL Format (JSON Lines):**
  ```json
  {"text": "Document text 1", "category": "Category A"}
  {"text": "Document text 2", "category": "Category B"}
  ```

### 2. Loading Data in Code
You can import and use the loader in your test scripts:
```python
from src.dataset.loader import load_dataset

# Load dataset (CSV, JSON, or JSONL)
dataset = load_dataset("data/your_dataset.csv")

# Retrieve raw document text strings
texts = dataset.get_texts()

# Retrieve ground truth labels for a specific aspect (e.g. 'category', 'sentiment')
true_labels = dataset.get_aspect_labels("category")
```

---

## 📈 Running the Evaluations & Demos

### 1. Interactive Streamlit Web Application (Recommended)
Launch the premium web UI to converse with the agent, view 2D PCA cluster maps in real-time, inspect individual cluster partitions/medoids, and load your custom dataset:
```bash
streamlit run app.py
```
Open the provided URL (typically `http://localhost:8501`) in your browser.
* **Configure Providers:** Easily toggle between Gemini, OpenAI, Anthropic, GitHub Models, or a local Ollama server in the sidebar.
* **Interactive Visualization:** The web app renders a dynamic, high-contrast Plotly 2D PCA map highlighting the 20 representative documents selected via Farthest Point Sampling.
* **Explore Partitions:** Click through detailed cluster breakdowns, inspect representative medoids, and view all items in each cluster in interactive tables.

### 2. Interactive Jupyter Dashboard
Open `notebooks/interactive_demo.ipynb` using Jupyter Notebook or VS Code. Run the cells to display the interactive UI. You can choose a dataset (e.g., Amazon Reviews) and type commands like *"group by Sentiment"* or *"group by Feature"* to see the clusters reorganize dynamically in real time.

### 3. Automated Simulation Experiments
Run the automated experiment comparing C3 against Traditional Active Learning (pairwise queries) and Unconstrained K-Means:
```bash
python -m src.run_experiments
```
This will run the simulation loop, compute performance metrics (Adjusted Rand Index vs. Interaction Cost Units), and save a comparison chart to `notebooks/experiments_plot.png`.

---

## 🧪 Running Unit Tests

Verify the correctness of the mathematical components (PCK-Means, constraint propagation, metric calculations, and pipeline mapping):
```bash
python -m pytest tests/
```