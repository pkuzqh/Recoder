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
<center class="third">
    <img src="https://github.com/FSE2021anonymous/Recoder/blob/master/picture/Insert.png" width="100"/>
    <img src="https://github.com/FSE2021anonymous/Recoder/blob/master/picture/Modify.png" width="100"/>
</center>

## Online Demo
A demo to show our model.
