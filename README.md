# Recoder
A Pytorch Implementation of "A Syntax-Guided Edits Decoder for Neural Program Repair"

![image](https://github.com/FSE2021anonymous/Recoder/blob/master/picture/overviewmodel.png)

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
## Two Examples of Edits

<img src="https://github.com/FSE2021anonymous/Recoder/blob/master/picture/Insert.png" width="480"/><img src="https://github.com/FSE2021anonymous/Recoder/blob/master/picture/Modify.png" width="480"/> 

## Gnerated Patches
The generated patches are in the floder [Result/](https://github.com/FSE2021anonymous/Recoder/blob/master/Result).

## Online Demo
[A demo to show Recoder.](http://123.57.129.161:8081/)
