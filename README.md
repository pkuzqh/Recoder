# Recoder
A PyTorch Implementation of "A Syntax-Guided Edits Decoder for Neural Program Repair"

![image](https://github.com/FSE2021anonymous/Recoder/blob/master/picture/overviewmodel.png.Jpeg)

## Dependency
* Python 3.7
* PyTorch 1.3
* Defects4J
* Java 8

## Training Set
The data will be published after the paper is accepted.
## Train a New Model
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

<img src="https://github.com/FSE2021anonymous/Recoder/blob/master/picture/Insert.png" metaname="viewport" width="400"/><img src="https://github.com/FSE2021anonymous/Recoder/blob/master/picture/Modify.png" metaname="viewport" width="400"/> 

## Gnerated Patches
The generated patches are in the folder [Result/](https://github.com/FSE2021anonymous/Recoder/blob/master/Result).

## Online Demo
[A demo to show Recoder.](http://35.194.10.109:8081/)
