import sys
import torch
import torch.utils.data as data
import random
import pickle
import os
from nltk import word_tokenize
from vocab import VocabEntry
import numpy as np
class SumDataset(data.Dataset):
    def __init__(self, config, dataName="train"):
        self.train_path = "DataProcess/train.txt"
        self.val_path = "DataProcess/test.txt"  # "validD.txt"
        self.test_path = "DataProcess/test.txt"
        self.Nl_Voc = {"pad": 0, "Unknown": 1}
        self.Code_Voc = {"pad": 0, "Unknown": 1}
        self.Char_Voc = {"pad": 0, "Unknown": 1}
        self.Nl_Len = config.NlLen
        self.Code_Len = config.CodeLen
        self.Char_Len = config.WoLen
        self.batch_size = config.batch_size
        self.PAD_token = 0
        self.data = None
        self.dataName = dataName
        self.Codes = []
        self.Nls = []
        if not os.path.exists("nl_sum_voc.pkl"):
            self.init_dic()
        self.Load_Voc()
        if dataName == "train":
            if os.path.exists("data_sum.pkl"):
                self.data = pickle.load(open("data_sum.pkl", "rb"))
                return
            self.data = self.preProcessData(open(self.train_path, "r", encoding='iso-8859-1'))
        elif dataName == "val":
            if os.path.exists("valdata_sum.pkl"):
                self.data = pickle.load(open("valdata_sum.pkl", "rb"))
                return
            self.data = self.preProcessData(open(self.val_path, "r", encoding='iso-8859-1'))
        else:
            if os.path.exists("testdata_sum.pkl"):
                self.data = pickle.load(open("testdata_sum.pkl", "rb"))
                return
            self.data = self.preProcessData(open(self.test_path, "r", encoding='iso-8859-1'))

    def Load_Voc(self):
        if os.path.exists("nl_sum_voc.pkl"):
            self.Nl_Voc = pickle.load(open("nl_sum_voc.pkl", "rb"))
        if os.path.exists("code_sum_voc.pkl"):
            self.Code_Voc = pickle.load(open("code_sum_voc.pkl", "rb"))
        if os.path.exists("char_sum_voc.pkl"):
            self.Char_Voc = pickle.load(open("char_sum_voc.pkl", "rb"))

    def init_dic(self):
        print("initVoc")
        f = open(self.train_path, "r", encoding='iso-8859-1')
        lines = f.readlines()
        maxNlLen = 0
        maxCodeLen = 0
        maxCharLen = 0
        Nls = []
        Codes = []
        for i in range(int(len(lines) / 2)):
            Code = lines[2 * i + 1].strip()
            Nl = lines[2 * i].strip()
            #if "^" in Nl
                #print(Nl)
            Nl_tokens = ["<start>"] + word_tokenize(Nl.lower()) + ["<end>"]
            Code_Tokens = Code.lower().split()
            Nls.append(Nl_tokens)
            # Nls.append(Code_Tokens)
            Codes.append(Code_Tokens)
            maxNlLen = max(maxNlLen, len(Nl_tokens))
            maxCodeLen = max(maxCodeLen, len(Code_Tokens))
        # print(Nls)
        # print("------------------")
        nl_voc = VocabEntry.from_corpus(Nls, size=50000, freq_cutoff=3)
        code_voc = VocabEntry.from_corpus(Codes, size=50000, freq_cutoff=3)
        self.Nl_Voc = nl_voc.word2id
        self.Code_Voc = code_voc.word2id

        for x in self.Nl_Voc:
            maxCharLen = max(maxCharLen, len(x))
            for c in x:
                if c not in self.Char_Voc:
                    self.Char_Voc[c] = len(self.Char_Voc)
        for x in self.Code_Voc:
            maxCharLen = max(maxCharLen, len(x))
            for c in x:
                if c not in self.Char_Voc:
                    self.Char_Voc[c] = len(self.Char_Voc)
        if "<start>" in self.Nl_Voc:
            print("right")
        print(len(self.Nl_Voc), len(self.Code_Voc))
        open("nl_sum_voc.pkl", "wb").write(pickle.dumps(self.Nl_Voc))
        open("code_sum_voc.pkl", "wb").write(pickle.dumps(self.Code_Voc))
        open("char_sum_voc.pkl", "wb").write(pickle.dumps(self.Char_Voc))
        #print(self.Nl_Voc)
        #print(self.Code_Voc)
        print(maxNlLen, maxCodeLen, maxCharLen)
    def Get_Em(self, WordList, NlFlag=True):
        ans = []
        for x in WordList:
            if NlFlag:
                if x not in self.Nl_Voc:
                    ans.append(1)
                else:
                    ans.append(self.Nl_Voc[x])
            else:
                if x not in self.Code_Voc:
                    ans.append(1)
                else:
                    ans.append(self.Code_Voc[x])
        return ans
    def Get_Char_Em(self, WordList):
        ans = []
        for x in WordList:
            tmp = []
            for c in x:
                c_id = self.Char_Voc[c] if c in self.Char_Voc else 1
                tmp.append(c_id)
            ans.append(tmp)
        return ans
    def pad_seq(self, seq, maxlen):
        act_len = len(seq)
        if len(seq) < maxlen:
            seq = seq + [self.PAD_token] * maxlen
            seq = seq[:maxlen]
        else:
            seq = seq[:maxlen]
            act_len = maxlen
        return seq, act_len
    def pad_list(self,seq, maxlen1, maxlen2):
        if len(seq) < maxlen1:
            seq = seq + [[self.PAD_token] * maxlen2] * maxlen1
            seq = seq[:maxlen1]
        else:
            seq = seq[:maxlen1]
        return seq
    def getAdMatrix(self, codetokens):
        lst = codetokens#codetokens.split()
        #print(codetokens)
        currNode = node(lst[0])
        currNode.id = 0
        nodedist = {}
        for i, x in enumerate(lst):
            if i == 0:
                nodedist[i] = currNode
                continue
            if not x[-1] == "^" or ("^" in x and "_" not in x):
                newNode = node(x)
                newNode.father = currNode
                currNode.child.append(newNode)
                newNode.id = i
                currNode = newNode
                nodedist[i] = newNode
            else:
                newNode = node(x)
                newNode.child.append(currNode)
                if currNode.father:
                    newNode.child.append(currNode.father)
                newNode.id = i
                nodedist[i] = newNode
                currNode.child.append(newNode)
                if currNode.father:
                    currNode.father.child.append(newNode)
                #print(x, currNode.name)
                currNode = currNode.father
        admatrix = []
        upbound = min(self.Code_Len, len(lst))
        for i in range(upbound):
            ids = []
            for x in nodedist[i].child:
                if x.id < self.Code_Len:
                    ids.append(x.id)
            ids.append(nodedist[i].id)
            if nodedist[i].father:
                if nodedist[i].father.id < self.Code_Len:
                    ids.append(nodedist[i].father.id)
            #tmp = np.sum(np.eye(len(lst))[ids])
            admatrix.append(ids)
        return admatrix



    def preProcessData(self, dataFile):
        lines = dataFile.readlines()
        Nl_Sentences = []
        Code_Sentences = []
        Nl_Chars = []
        Code_Chars = []
        admatrix = []
        res = []
        from tqdm import tqdm
        for i in tqdm(range(int(len(lines) / 2))):
            code = lines[2 * i + 1].strip()
            nl = lines[2 * i].strip()
            code_tokens = code.lower().split()
            try:
                admatrix.append(self.getAdMatrix(code_tokens))
            except:
                continue
            nl_tokens = ["<start>"] + word_tokenize(nl.lower()) + ["<end>"]
            Code_Sentences.append(self.Get_Em(code_tokens, False))
            Nl_Sentences.append(self.Get_Em(nl_tokens))
            Nl_Chars.append(self.Get_Char_Em(nl_tokens))
            Code_Chars.append(self.Get_Char_Em(code_tokens))
            #admatrix.append(self.getAdMatrix(code_tokens))
            res.append(Nl_Sentences[-1][1:])
        for i in range(len(Nl_Sentences)):
            Nl_Sentences[i], _ = self.pad_seq(Nl_Sentences[i], self.Nl_Len)
            Code_Sentences[i], _ = self.pad_seq(Code_Sentences[i], self.Code_Len)
            res[i], _ = self.pad_seq(res[i], self.Nl_Len)
            for j in range(len(Nl_Chars[i])):
                Nl_Chars[i][j], _ = self.pad_seq(Nl_Chars[i][j], self.Char_Len)
            for j in range(len(Code_Chars[i])):
                Code_Chars[i][j], _ = self.pad_seq(Code_Chars[i][j], self.Char_Len)
            Nl_Chars[i] = self.pad_list(Nl_Chars[i], self.Nl_Len, self.Char_Len)
            Code_Chars[i] = self.pad_list(Code_Chars[i], self.Code_Len, self.Char_Len)
        batchs = [Nl_Sentences, Nl_Chars, Code_Sentences, Code_Chars, admatrix, res]
        batchs = np.array(batchs)
        self.data = batchs
        if self.dataName == "train":
            open("data_sum.pkl", "wb").write(pickle.dumps(batchs, protocol=4))
        if self.dataName == "val":
            open("valdata_sum.pkl", "wb").write(pickle.dumps(batchs, protocol=4))
        if self.dataName == "test":
            open("testdata_sum.pkl", "wb").write(pickle.dumps(batchs))
        return batchs



    def __getitem__(self, offset):
        ans = []
        for i in range(len(self.data)):
            if i == 4:
                tmp = []
                for j in range(len(self.data[i][offset])):
                    #print(np.sum(np.eye(self.Code_Len)[self.data[i][offset][j]], axis=0))
                    tmp.append(np.sum(np.eye(self.Code_Len)[self.data[i][offset][j]], axis=0))
                tmp = self.pad_list(tmp, self.Code_Len, self.Code_Len)
                tmp = np.array(tmp)
                tmp = tmp.reshape(1, self.Code_Len, self.Code_Len)
                ans.append(tmp)
            else:
                ans.append(np.array(self.data[i][offset]))
        return ans
    def __len__(self):
        return len(self.data[0])
class node:
    def __init__(self, name):
        self.name = name
        self.father = None
        self.child = []
        self.id = -1
