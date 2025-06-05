# UrbanGraphNet: GNN-Transformer for City Layout Modeling and Generation 🏙️🧠

UrbanGraphNet is a novel approach to modeling city infrastructure using graphs. We represent urban road networks as graphs where intersections are nodes and roads are edges. Our custom hybrid architecture combines Graph Attention Networks (GAT) with Transformer mechanisms to learn spatial-temporal patterns and predict layout extensions such as new intersections or road structures.

## 🚀 Overview

- 🔗 **Nodes**: Road intersections
- ↔️ **Edges**: Roads, carrying attributes like road type, congestion, and directionality
- 🔮 **Goal**: Predict future expansions or optimize existing layouts by generating nodes and suggesting new connections
- 🧠 **Model**: GAT + Transformer hybrid for learning both local (GAT) and global (Transformer) structure

---

## 🛠 Architecture Highlights

### 🔹 Graph Neural Network Base
- Uses **Graph Attention Networks (GAT)** to learn neighborhood-aware embeddings
- Integrates **attention coefficients** for weighted message passing

### 🔹 Transformer Layer (Custom)
- Global context integration using **multi-head attention**
- Embeds spatial priors and temporal road usage patterns
- Enhances feature interactions beyond k-hop neighborhoods

### 🔹 Node Generation Logic
- Inserts new nodes based on:
  - Traffic congestion patterns
  - Road type (e.g., highway, residential)
  - Positional geometry
  - Local degree of connectivity

### 🔹 Features Considered
- Congestion levels (as dynamic weights)
- Road width, type, and curvature
- Temporal data if available (e.g., rush-hour congestion)

---

## 🧪 Functionalities

- 🔍 **Layout Prediction**: Learns to predict plausible city structures from existing topologies
- 🆕 **Node Insertion**: Generates new intersections/roads based on learned features
- 🗺️ **Spatial Reasoning**: Accounts for geometric and contextual relevance of road networks
- 📈 **Training Pipeline**: Includes loss metrics based on graph connectivity and prediction accuracy

---

## ⚙️ Dependencies

```bash
pip install torch torch-geometric networkx matplotlib scikit-learn
```
> **Note:** You may also need pyg-lib and torch-scatter based on your platform for GNNs.

---
## 🏁 How to Run
🔹 1. Load City Graph
```python
import networkx as nx
G = nx.read_gpickle("data/city_graph_sample.gpickle")
```
🔹 2. Train the Model
```python
from models.urban_gnn import UrbanGraphNet
model = UrbanGraphNet(...)
model.train(G)
```
🔹 3. Predict/Generate
```python
pred = model.predict(G)
visualize_prediction(pred)
```
---

## 📊 Metrics
- Graph Edit Distance (GED)

- Node Addition Accuracy

- Edge Precision / Recall

- Connectivity Preservation Score

- Spatial Distribution Consistency

---
## 🌆 Visual Output
Before: Existing road layout

After: Suggested expansions, inserted intersections, predicted changes

Use the built-in `visualize_prediction()` function or `matplotlib + networkx` to render the graph overlays.

---

## 📚 Future Work
- Add support for multi-city training datasets

- Introduce reinforcement learning for feedback-based layout generation

- Combine satellite imagery + GNN predictions for validation

- Deploy as web-based interactive layout suggestion tool

---

## 👨‍💻 Authors
**Developed by Lovy Verma**
---

## 📝 License
**MIT License**
---

## 🙏 Acknowledgements
Torch Geometric (PyG)

OpenStreetMap & city data sources

PyTorch, NetworkX, and contributors to open graph learning tools


---
