# SHDF

## Environment
The codes of SHDF are implemented and tested under the following environment:

python = 3.8.20

torch = 2.4.0

numpy = 1.24.3

scipy = 1.10.1

## Datasets
We follow the paper "Knowledge Graph Self-Supervised Rationalization
for Recommendation" to process data

## Training

- Last-FM dataset
```bash 
python main.py --dataset last-fm --epochs 200 --lr 0.0001 --latdim 64
```
- Mind-f dataset
```bash 
python main.py --dataset mind-f --epochs 100 --lr 0.001 --latdim 64
```
- alibaba-fashion dataset
```bash 
python main.py --dataset last-fm --epochs 250 --lr 0.001 --latdim 64
```