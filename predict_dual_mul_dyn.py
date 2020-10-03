import argparse
import os
import pickle
import random
import sys

import numpy as np
import tensorflow as tf

from myutils import batch_gen, init_tf, seq2sent
import keras
import keras.backend as K
from model import create_model
from timeit import default_timer as timer
from graphlayer import GCNLayer
import pandas as pd

def gen_pred(model, data, comstok, comlen, batchsize, config, strat='greedy'):
    tdats, coms, wsmlnodes, wedge_1 = zip(*data.values())
    tdats = np.array(tdats)
    coms = np.array(coms)
    wsmlnodes = np.array(wsmlnodes)
    wedge_1 = np.array(wedge_1)

    for i in range(1, comlen):
        results = model.predict([tdats, coms, wsmlnodes, wedge_1],
                                batch_size=batchsize)
        for c, s in enumerate(results):
            coms[c][i] = np.argmax(s)

    final_data = {}
    for fid, com in zip(data.keys(), coms):
        final_data[fid] = seq2sent(com, comstok)

    return final_data

def gendescr_4inp(model, data, comstok, comlen, batchsize, config, strat='greedy'):

    tdats,sdats, coms, wsmlnodes, wedge_1 = zip(*data.values())
    tdats = np.array(tdats)
    sdats = np.array(sdats)
    coms = np.array(coms)
    wsmlnodes = np.array(wsmlnodes)
    wedge_1 = np.array(wedge_1)

    for i in range(1, comlen):
        results = model.predict([tdats, sdats, coms, wsmlnodes, wedge_1], batch_size=batchsize)
        for c, s in enumerate(results):
            coms[c][i] = np.argmax(s)

    final_data = {}
    for fid, com in zip(data.keys(), coms):
        final_data[fid] = seq2sent(com, comstok)

    return final_data

def gen_method(model, data, comstok, comlen, batchsize, config, strat='greedy',hightoken=None):

    musttoken = np.asarray([0, 75435, 75436, 74999], dtype=np.int32)
    newvoacabsize = 5000
    dynamicsize = 400

    tdats, sdats, coms, wsmlnodes, wedge_1, mets = zip(*data.values())
    tdats = np.array(tdats)
    sdats = np.array(sdats)
    coms = np.array(coms)
    wsmlnodes = np.array(wsmlnodes)
    wedge_1 = np.array(wedge_1)
    mets = np.array(mets)

    dynamic_token = pd.unique(np.concatenate(
        (coms.flatten(), mets.flatten(), tdats.flatten(), wsmlnodes.flatten(),
         sdats.flatten()),
        axis=0))

    token = pd.unique(np.concatenate(
        (musttoken, hightoken[:newvoacabsize-dynamicsize-4], dynamic_token),  # 4 3996+650 350
        axis=0))

    if len(token) < newvoacabsize:
        token = np.concatenate(
            (token, hightoken[newvoacabsize-dynamicsize-4:]),
            axis=0)
    else:
        token = token[:newvoacabsize]
    k2v = dict(enumerate(pd.unique(token)[:newvoacabsize]))
    v2k = dict(zip(k2v.values(), k2v.keys()))

    tdats = np.where(np.isin(tdats, np.array(list(v2k.keys()))), tdats, 74999)
    sdats = np.where(np.isin(sdats, np.array(list(v2k.keys()))), sdats, 74999)
    wsmlnodes = np.where(np.isin(wsmlnodes, np.array(list(v2k.keys()))), wsmlnodes, 74999)
    mets = np.where(np.isin(mets, np.array(list(v2k.keys()))), mets, 74999)
    coms = np.where(np.isin(coms, np.array(list(v2k.keys()))), coms, 74999)
    tdats = np.vectorize(v2k.get)(tdats)
    sdats = np.vectorize(v2k.get)(sdats)
    wsmlnodes = np.vectorize(v2k.get)(wsmlnodes)
    mets = np.vectorize(v2k.get)(mets)
    coms = np.vectorize(v2k.get)(coms)

    for i in range(1, comlen):
        results = model.predict([tdats, sdats, coms, wsmlnodes, wedge_1,mets], batch_size=batchsize)
        for c, s in enumerate(results):
            coms[c][i] = np.argmax(s)

    coms = np.vectorize(k2v.get)(coms)

    final_data = {}
    for fid, com in zip(data.keys(), coms):
        final_data[fid] = seq2sent(com, comstok)

    return final_data

def write_predict(index):
    config = dict()
    config['maxastnodes'] = 100
    config['asthops'] = 2
    allfids = list(seqdata['ctest'].keys())
    datvocabsize = 5000
    comvocabsize = 5000
    smlvocabsize = 5000
    config['tdatvocabsize'] = datvocabsize
    config['comvocabsize'] = comvocabsize
    config['smlvocabsize'] = smlvocabsize
    config['tdatlen'] = 50
    config['comlen'] = len(list(seqdata['ctrain'].values())[0])
    config['smllen'] = len(list(seqdata['strain_nodes'].values())[0])
    config['metlen'] = 17
    config['batch_size'] = batchsize

    comlen = len(seqdata['ctest'][list(seqdata['ctest'].keys())[0]])
    config, model = create_model(modeltype, config)
    print('config',config)
    print("MODEL LOADED")
    model.load_weights(modelfile[:-5]+index+'.h5')

    node_data = seqdata['stest_nodes']
    edgedata = seqdata['stest_edges']
    met_test = pickle.load(open('{}/mtest.pkl'.format(dataprep), 'rb'))
    com_test = pickle.load(open('{}/com_test.pkl'.format(dataprep), 'rb'))
    config['batch_maker'] = 'gnn_method'
    print(model.summary())

    comstart = np.zeros(comlen)
    stk = comstok.w2i['<s>']
    comstart[0] = stk
    outfn = outdir + "predictions/predict-codegnndualfcinfo_E01.txt"[:-6]+index+".txt"
    outf = open(outfn, 'w')
    print("writing to file: " + outfn)
    batch_sets = [allfids[i:i + batchsize] for i in range(0, len(allfids), batchsize)]

    for c, fid_set in enumerate(batch_sets):
        st = timer()
        for fid in fid_set:
            com_test[fid] = comstart

        bg = batch_gen(seqdata, 'test', config, nodedata=node_data, edgedata=edgedata,metdata=met_test,comdata=com_test,hightoken=hightoken)
        batch = bg.make_batch(fid_set)
        if modeltype == 'codegnndualfcinfo':
            batch_results = gen_method(model, batch, comstok, comlen, batchsize, config, strat='greedy',hightoken=hightoken)
        elif modeltype == 'codegnndualfc':
            batch_results = gendescr_4inp(model, batch, comstok, comlen, batchsize, config, strat='greedy',hightoken=hightoken)
        elif modeltype == 'codegnngru':
            batch_results = gen_pred(model, batch, comstok, comlen, batchsize, config, strat='greedy',hightoken=hightoken)
        else:
            batch_results = gen_method(model, batch, comstok, comlen, batchsize, config, strat='greedy',
                                       hightoken=hightoken)
        for key, val in batch_results.items():
            outf.write("{}\t{}\n".format(key, val))

        end = timer()
        print("{} processed, {} per second this batch".format((c + 1) * batchsize, int(batchsize / (end - st))),
              end='\r')

    outf.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model', type=str, default='modelout/models/codegnndualfcinfo_E07.h5')
    parser.add_argument('--modeltype', dest='modeltype', type=str, default='codegnndualfcinfo')
    parser.add_argument('--gpu', dest='gpu', type=str, default='')
    parser.add_argument('--data', dest='dataprep', type=str, default='../codesum/data')
    parser.add_argument('--outdir', dest='outdir', type=str, default='modelout/')
    parser.add_argument('--batch-size', dest='batchsize', type=int, default=128)
    parser.add_argument('--outfile', dest='outfile', type=str, default=None)
    args = parser.parse_args()

    modelfile = args.model
    outdir = args.outdir
    dataprep = args.dataprep
    gpu = args.gpu
    batchsize = args.batchsize
    modeltype = args.modeltype
    outfile = args.outfile

    if modeltype == None:
        modeltype = modelfile.split('_')[0].split('/')[-1][:-3]
        print('modeltype', modeltype)
    print(modeltype)
    print('load_tok')
    hightoken = pickle.load(open('{}/hightoken.np'.format(dataprep), 'rb'), encoding='UTF-8')
    tdatstok = pickle.load(open('{}/tdatstok.pkl'.format(dataprep), 'rb'), encoding='UTF-8')
    comstok = tdatstok
    smltok = tdatstok

    print('load dataset.pkl')
    seqdata = pickle.load(open('%s/dataset.pkl' % (dataprep), 'rb'))
    indexs = ['01', '02', '03', '04', '05', '06', '07', '08', '09']
    for index in indexs:
        write_predict(index)




