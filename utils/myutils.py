import random
import sys
import tensorflow as tf
import keras
import numpy as np
import pandas as pd

start = 0
end = 0

# init gpu
def init_tf(gpu):
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

# the index corresponds to the word one to one
def index2word(tok):
    i2w = {}
    for word, index in tok.w2i.items():
        i2w[index] = word
    return i2w

# converts the index sequence to the corresponding sentence
def seq2sent(seq, tokenizer):
    sent = []
    check = index2word(tokenizer)
    for i in seq:
        sent.append(check[i])
    return (' '.join(sent))

# converts the index sequence to the corresponding token list
def seq2tokenlist(seq, tokenizer):
    sent = []
    check = index2word(tokenizer)
    for i in seq:
        sent.append(check[i])
    return sent

# construct batch data
class batch_gen(keras.utils.Sequence):
    def __init__(self, seqdata, tt, config, nodedata=None, edgedata=None, metdata=None, comdata=None, hightoken=None):
        self.comvocabsize = config['comvocabsize']
        self.tt = tt
        self.batch_size = config['batch_size']
        self.seqdata = seqdata
        self.allfids = list(seqdata['dt%s' % (tt)].keys())
        self.config = config
        self.edgedata = edgedata
        self.nodedata = nodedata
        self.metdata = metdata # method name keywords
        self.comdata = comdata
        self.hightoken = hightoken  # High-frequency tokens for dynamic vocabulary
        self.musttoken = np.asarray([0, 75435, 75436, 74999], dtype=np.int32)
        self.newvoacabsize = 5000 # dynamic vocabulary size
        self.dynamicsize = 400 # data vocabulary size, code token + summary token
        random.shuffle(self.allfids)

    def __getitem__(self, idx):
        start = (idx * self.batch_size)
        end = self.batch_size * (idx + 1)
        batchfids = self.allfids[start:end]
        return self.make_batch(batchfids)

    def make_batch(self, batchfids):
        if self.config['batch_maker'] == 'gnn_method':
            return self.divideseqs_gnn_method(batchfids, self.seqdata, self.nodedata, self.edgedata, self.metdata,
                                              self.comdata, self.comvocabsize, self.tt)

    def __len__(self):
        return int(np.ceil(len(list(self.seqdata['dt%s' % (self.tt)])) / self.batch_size))

    def on_epoch_end(self):
        random.shuffle(self.allfids)

    # use dynamic vocabulary
    def divideseqs_gnn_method(self, batchfids, seqdata, nodedata, edge1, metdata, comdata, comvocabsize, tt):
        import keras.utils

        tdatseqs = list()
        sdatseqs = list()
        comseqs = list()
        smlnodes = list()
        wedge_1 = list()
        metseqs = list()
        fiddat = dict()
        comout_pres = list()

        for fid in batchfids:

            wtdatseq = seqdata['dt%s' % (tt)][fid]
            wsdatseq = seqdata['ds%s' % (tt)][fid]
            wcomseq = comdata[fid]
            wmetseq = metdata[fid]
            try:
                wsmlnodes = nodedata[fid]
            except:
                continue
            try:
                edge_1 = edge1[fid]
            except:
                continue

            wsmlnodes = wsmlnodes[:self.config['maxastnodes']]
            tmp = np.zeros(self.config['maxastnodes'], dtype='int32')
            tmp[:wsmlnodes.shape[0]] = wsmlnodes
            wsmlnodes = np.int32(tmp)

            edge_1 = np.asarray(edge_1.todense())
            edge_1 = edge_1[:self.config['maxastnodes'], :self.config['maxastnodes']]
            tmp_1 = np.zeros((self.config['maxastnodes'], self.config['maxastnodes']), dtype='int32')
            tmp_1[:edge_1.shape[0], :edge_1.shape[1]] = edge_1
            edge_1 = np.int32(tmp_1)

            wtdatseq = wtdatseq[:self.config['tdatlen']]
            newlen = self.config['sdatlen'] - len(wsdatseq)
            if newlen < 0:
                newlen = 0
            wsdatseq = wsdatseq.tolist()
            for k in range(newlen):
                wsdatseq.append(np.zeros(self.config['stdatlen']))
            for i in range(0, len(wsdatseq)):
                wsdatseq[i] = np.array(wsdatseq[i])[:self.config['stdatlen']]
            wsdatseq = np.asarray(wsdatseq)
            wsdatseq = wsdatseq[:self.config['sdatlen'], :]

            if tt == 'test':
                fiddat[fid] = [wtdatseq, wsdatseq, wcomseq, wsmlnodes, edge_1, wmetseq]
            else:
                for i in range(0, len(wcomseq)):
                    tdatseqs.append(wtdatseq)
                    sdatseqs.append(wsdatseq)
                    smlnodes.append(wsmlnodes)
                    wedge_1.append(edge_1)
                    metseqs.append(wmetseq)

                    comseq = wcomseq[0:i]
                    comout_pre = wcomseq[i]

                    for j in range(0, len(wcomseq)):
                        try:
                            comseq[j]
                        except IndexError as ex:
                            comseq = np.append(comseq, 0)

                    comseqs.append(comseq)
                    comout_pres.append(comout_pre)

        tdatseqs = np.asarray(tdatseqs)
        sdatseqs = np.asarray(sdatseqs)
        smlnodes = np.asarray(smlnodes)

        wedge_1 = np.asarray(wedge_1)
        metseqs = np.asarray(metseqs)
        comseqs = np.asarray(comseqs)

        if tt == 'test':
            # The dynamic vocabulary for testing is built in predict file
            return fiddat
        else:
            # creat dynamic vocabulary

            # input token and summary token, the number is about 1,200 to 1,300,
            dynamic_token = pd.unique(np.concatenate(
                (comseqs.flatten(), comout_pres, metseqs.flatten(), tdatseqs.flatten(), smlnodes.flatten(),
                 sdatseqs.flatten()),
                axis=0))
            # joint high frequency token
            # excluding high-frequency tokens, the number of input tokens is about 200-332, we choose 400
            token = pd.unique(np.concatenate(
                (self.musttoken, self.hightoken[:self.newvoacabsize-self.dynamicsize-4], dynamic_token),  # 4 4596 400
                axis=0))
            if len(token) < self.newvoacabsize:
                token = np.concatenate(
                    (token, self.hightoken[self.newvoacabsize-self.dynamicsize-4:]),
                    axis=0)
            else:
                token = token[:self.newvoacabsize]
            k2v = dict(enumerate(pd.unique(token)[:self.newvoacabsize]))
            v2k = dict(zip(k2v.values(), k2v.keys()))
            # reconstruct input data
            tdatseqs = np.where(np.isin(tdatseqs, np.array(list(v2k.keys()))), tdatseqs, 74999)
            sdatseqs = np.where(np.isin(sdatseqs, np.array(list(v2k.keys()))), sdatseqs, 74999)
            smlnodes = np.where(np.isin(smlnodes, np.array(list(v2k.keys()))), smlnodes, 74999)
            metseqs = np.where(np.isin(metseqs, np.array(list(v2k.keys()))), metseqs, 74999)
            comseqs = np.where(np.isin(comseqs, np.array(list(v2k.keys()))), comseqs, 74999)
            comout_pres = np.where(np.isin(comout_pres, np.array(list(v2k.keys()))), comout_pres, 74999)
            tdatseqs = np.vectorize(v2k.get)(tdatseqs)
            sdatseqs = np.vectorize(v2k.get)(sdatseqs)
            smlnodes = np.vectorize(v2k.get)(smlnodes)
            metseqs = np.vectorize(v2k.get)(metseqs)
            comseqs = np.vectorize(v2k.get)(comseqs)
            comout_pres = np.vectorize(v2k.get)(comout_pres)
            comouts = keras.utils.to_categorical(comout_pres, num_classes=self.newvoacabsize)

            return [[tdatseqs, sdatseqs, comseqs, smlnodes, wedge_1, metseqs],
                    comouts]

    # use all vocabulary
    def divideseqs_gnn_method(self, batchfids, seqdata, nodedata, edge1, metdata, comdata, comvocabsize, tt):
        import keras.utils

        tdatseqs = list()
        sdatseqs = list()
        comseqs = list()
        smlnodes = list()
        wedge_1 = list()
        metseqs = list()
        fiddat = dict()
        comout_pres = list()

        for fid in batchfids:

            wtdatseq = seqdata['dt%s' % (tt)][fid]
            wsdatseq = seqdata['ds%s' % (tt)][fid]
            wcomseq = comdata[fid]
            wmetseq = metdata[fid]
            try:
                wsmlnodes = nodedata[fid]
            except:
                continue
            try:
                edge_1 = edge1[fid]
            except:
                continue

            wsmlnodes = wsmlnodes[:self.config['maxastnodes']]
            tmp = np.zeros(self.config['maxastnodes'], dtype='int32')
            tmp[:wsmlnodes.shape[0]] = wsmlnodes
            wsmlnodes = np.int32(tmp)

            edge_1 = np.asarray(edge_1.todense())
            edge_1 = edge_1[:self.config['maxastnodes'], :self.config['maxastnodes']]
            tmp_1 = np.zeros((self.config['maxastnodes'], self.config['maxastnodes']), dtype='int32')
            tmp_1[:edge_1.shape[0], :edge_1.shape[1]] = edge_1
            edge_1 = np.int32(tmp_1)

            wtdatseq = wtdatseq[:self.config['tdatlen']]
            newlen = self.config['sdatlen'] - len(wsdatseq)
            if newlen < 0:
                newlen = 0
            wsdatseq = wsdatseq.tolist()
            for k in range(newlen):
                wsdatseq.append(np.zeros(self.config['stdatlen']))
            for i in range(0, len(wsdatseq)):
                wsdatseq[i] = np.array(wsdatseq[i])[:self.config['stdatlen']]
            wsdatseq = np.asarray(wsdatseq)
            wsdatseq = wsdatseq[:self.config['sdatlen'], :]

            if tt == 'test':
                fiddat[fid] = [wtdatseq, wsdatseq, wcomseq, wsmlnodes, edge_1, wmetseq]
            else:
                for i in range(0, len(wcomseq)):
                    tdatseqs.append(wtdatseq)
                    sdatseqs.append(wsdatseq)
                    smlnodes.append(wsmlnodes)
                    wedge_1.append(edge_1)
                    metseqs.append(wmetseq)

                    comseq = wcomseq[0:i]
                    comout_pre = wcomseq[i]

                    for j in range(0, len(wcomseq)):
                        try:
                            comseq[j]
                        except IndexError as ex:
                            comseq = np.append(comseq, 0)
                    comseqs.append(comseq)
                    comout_pres.append(comout_pre)

        tdatseqs = np.asarray(tdatseqs)
        sdatseqs = np.asarray(sdatseqs)
        smlnodes = np.asarray(smlnodes)

        wedge_1 = np.asarray(wedge_1)
        metseqs = np.asarray(metseqs)
        comseqs = np.asarray(comseqs)

        if tt == 'test':
            return fiddat
        else:
            comouts = keras.utils.to_categorical(comout_pres, num_classes=comvocabsize)

            return [[tdatseqs, sdatseqs, comseqs, smlnodes, wedge_1, metseqs],
                    comouts]
