# Recoder
A PyTorch Implementation of "A Syntax-Guided Edits Decoder for Neural Program Repair"

# Introduction
Automated Program Repair (APR) helps improve the efficiency of software development and maintenance. Recent APR techniques use deep learning, particularly the encoder-decoder architecture, to generate patches. Though existing DL-based APR approaches have proposed different encoder architectures, the decoder remains to be the standard one, which generates a sequence of tokens one by one to replace the faulty statement. This decoder has multiple limitations: 1) allowing to generate syntactically incorrect programs, 2) inefficiently representing small edits, and 3) not being able to generate project-specific identifiers.

In this paper, we propose Recoder, a syntax-guided edit decoder with placeholder generation. Recoder is novel in multiple aspects: 1) Recoder generates edits rather than modified code, allowing efficient representation of small edits; 2) Recoder is syntax-guided, with the novel provider/decider architecture to ensure the syntactic correctness of the patched program and accurate generation; 3) Recoder generates placeholders that could be instantiated as project-specific identifiers later.

We conduct experiments to evaluate Recoder on 395 bugs from Defects4J v1.2 and 420 additional bugs from Defects4J v2.0. Our results show that Recoder repairs 53 bugs on Defects4J v1.2, which achieves 26.2% improvement over the previous state-of-the-art approach for single-hunk bugs (TBar). Importantly, to our knowledge, Recoder is the first DL-based APR approach that has outperformed the traditional APR approaches on this dataset. Furthermore, Recoder also repairs 19 bugs on the additional bugs from Defects4J v2.0, which is 137.5% more than TBar (8 bugs) and 850% more than SimFix (2 bugs). This result suggests that Recoder has better generalizability than existing APR approaches.

![image](https://github.com/FSE2021anonymous/Recoder/blob/master/picture/overviewmodel.png.Jpeg)

# The Main File Tree of Recoder

```
.
├── Result
│   └── out
├── Picture
│   ├── Insert.png
│   ├── Modify.png
│   └── overviewmodel.png.Jpeg
├── Attention.py
├── Dataset.py
├── run.py
├── testDefect4j.py
├── totalrepair.py
└── Model.py
```


## Training Set
The data will be published after the paper is accepted.

## Train a New Model
```python
python3 run.py train
```
The saved model is ```checkpointSearch/best_model.ckpt```.

After our model was trained, we can 

## Generate Patches for Defects4J by
```python
python3 testDefects4j.py bugid
```

The generated patches are in folder ```patch/``` in json.

We further test the generated patches based on the test cases by

## Validate Patches
```python
python3 repair.py bugid
```

The results are in folder ```patches/``` in json.

## Two Examples of Edits

<img src="https://github.com/FSE2021anonymous/Recoder/blob/master/picture/Insert.png" metaname="viewport" width="400"/><img src="https://github.com/FSE2021anonymous/Recoder/blob/master/picture/Modify.png" metaname="viewport" width="400"/> 

## Gnerated Patches
The generated patches are in the folder [Result/](https://github.com/FSE2021anonymous/Recoder/blob/master/Result).


## Dependency
* Python 3.7
* PyTorch 1.3
* Defects4J
* Java 8

## Online Demo
[A demo to show Recoder.](http://35.194.10.109:8081/)
