# 📖 A Spectral Heterogeneous Diffusion Framework for Knowledge-aware Recommendation

## ⚙️ Environment
The codes of **SHDF** are implemented and tested under the following environment:

- Python = 3.8.20  
- PyTorch = 2.4.0  
- NumPy = 1.24.3  
- SciPy = 1.10.1  

For a complete setup, you can install all dependencies with:  
```bash
pip install -r requirements.txt
```

## 📂 Datasets

We follow the paper *"Knowledge Graph Self-Supervised Rationalization for Recommendation"* to preprocess the datasets.

## 🚀 Training

We provide training commands for three benchmark datasets:

- **Last-FM**

```
python main.py --dataset last-fm --epochs 200 --lr 0.0001 --latdim 64
```

- **Mind-F**

```
python main.py --dataset mind-f --epochs 100 --lr 0.001 --latdim 32
```

- **Alibaba-Fashion**

```
python main.py --dataset alibaba-fashion --epochs 250 --lr 0.001 --latdim 64
```

## 📑 Supplementary Material

We also provide supplementary materials containing:

- Detailed proofs of the formulas presented in the paper
- Time and space complexity analysis of the proposed framework

