# Recoder
A Pytorch Implementation of "A Syntax-Guided Edits Decoder for Neural Program Repair"

## Dependency
* python 3.7
* Pytorch 1.3
* Defects4J
* Java 8

## Raw Data
The data will be published after the paper is accepted.
## Train a new model
```python
python3 run.py train
```
## Generate Patches for Defects4J
```python
python3 testDefects4j.py bugid
```
## Validate Patches
```python
python3 repair.py bugid
```
## Online Demo
A demo to show our model.
