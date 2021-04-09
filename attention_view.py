import argparse
import os
import pickle
import random
import sys
import numpy as np
import tensorflow as tf
from keras.models import Model
from utils.myutils import batch_gen, init_tf, seq2sent,seq2tokenlist
import keras
import keras.backend as K
from utils.model import create_model
from timeit import default_timer as timer
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

def gen_method(model, data, comstok, comlen, batchsize, config, strat='greedy',hightoken=None):
    musttoken = np.asarray([0, 75435, 75436, 74999], dtype=np.int32)
    newvoacabsize = 5000
    dynamicsize = 400

    tdats, sdats, coms, wsmlnodes, wedge_1, mets = zip(*data.values())

    for fid, tdat,sdat,wsmlnode,met in zip(data.keys(), tdats, sdats, wsmlnodes, mets):
        print('code id',fid)
        tdattoken = seq2tokenlist(tdat, tdatstok)
        for k in range(sdat.shape[0]):
            print(k+1,seq2tokenlist(sdat[k], tdatstok))
        sdattoken = list(range(1,11))
        wsmlnodetoken = seq2tokenlist(wsmlnode, tdatstok)
        mettoken = seq2tokenlist(met, tdatstok)

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

    all1 = []
    all2 = []
    all3 = []
    all4 = []
    for i in range(1, comlen):

        activation_1_model = Model(inputs=model.input,
                               outputs=model.get_layer('activation_1').output)
        attention_1 = activation_1_model.predict([tdats, sdats, coms, wsmlnodes, wedge_1,mets], batch_size=batchsize)
        activation_2_model = Model(inputs=model.input,
                                   outputs=model.get_layer('activation_2').output)
        attention_2 = activation_2_model.predict([tdats, sdats, coms, wsmlnodes, wedge_1, mets], batch_size=batchsize)
        activation_3_model = Model(inputs=model.input,
                                   outputs=model.get_layer('activation_3').output)
        attention_3 = activation_3_model.predict([tdats, sdats, coms, wsmlnodes, wedge_1, mets], batch_size=batchsize)
        activation_4_model = Model(inputs=model.input,
                                   outputs=model.get_layer('activation_4').output)
        attention_4 = activation_4_model.predict([tdats, sdats, coms, wsmlnodes, wedge_1, mets], batch_size=batchsize)

        attention_1 = np.squeeze(attention_1)
        attention_2 = np.squeeze(attention_2)
        attention_3 = np.squeeze(attention_3)
        attention_4 = np.squeeze(attention_4)


        plot_attention(attention_1, mettoken, list(range(1,14)),'met'+str(i),fid)
        plot_attention(attention_2, tdattoken, list(range(1,14)),'tdat'+str(i),fid)
        plot_attention(attention_3, sdattoken, list(range(1,14)),'sdat'+str(i),fid)
        plot_attention(attention_4, wsmlnodetoken, list(range(1,14)),'node'+str(i),fid,7)

        mean1 = np.mean(attention_1, 0)
        mean2 = np.mean(attention_2, 0)
        mean3 = np.mean(attention_3, 0)
        mean4 = np.mean(attention_4, 0)
        all1.append(mean1)
        all2.append(mean2)
        all3.append(mean3)
        all4.append(mean4)

        results = model.predict([tdats, sdats, coms, wsmlnodes, wedge_1, mets], batch_size=batchsize)
        for c, s in enumerate(results):
            coms[c][i] = np.argmax(s)

    # map dynamic vocabulary token to all vocabulary token
    coms = np.vectorize(k2v.get)(coms)

    final_data = {}
    for fid, com in zip(data.keys(), coms):
        final_data[fid] = seq2sent(com, comstok)
        predict_sum = seq2tokenlist(com,comstok)
        plot_attention(np.array(all1), mettoken, predict_sum[1:],'met_all',fid)
        plot_attention(np.array(all2), tdattoken, predict_sum[1:],'tdat_all',fid)
        plot_attention(np.array(all3), sdattoken, predict_sum[1:],'sdat_all',fid)
        plot_attention(np.array(all4), wsmlnodetoken, predict_sum[1:],'node_all',fid,7)

    return final_data

# function for plotting the attention weights
def plot_attention(attention, sentence, predicted_sentence,name,fid,fontsize=10):
  fig = plt.figure(figsize=(10,10))
  ax = fig.add_subplot(1, 1, 1)
  ax.matshow(attention, cmap='viridis')

  fontdict = {'fontsize': fontsize}

  ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
  ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

  ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
  ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
  if not os.path.exists('./graph_'+str(fid)):
      os.mkdir('./graph_'+str(fid))
  plt.savefig('./graph_'+str(fid)+ '/' + str(name) + '.png')
  plt.show()

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

    ids = [34761729,12534240,28952314]
    for id in ids:
        # set up prediction string and output file
        comstart = np.zeros(comlen)
        stk = comstok.w2i['<s>']
        comstart[0] = stk

        i=allfids.index(id)
        batch_sets = [allfids[i:i + batchsize]]
        for c, fid_set in enumerate(batch_sets):
            st = timer()
            for fid in fid_set:
                com_test[fid] = comstart

            bg = batch_gen(seqdata, 'test', config, nodedata=node_data, edgedata=edgedata,metdata=met_test,comdata=com_test,hightoken=hightoken)
            batch = bg.make_batch(fid_set)

            if modeltype == 'codegnndualfcinfo':
                batch_results = gen_method(model, batch, comstok, comlen, batchsize, config, strat='greedy',hightoken=hightoken)

            for key, val in batch_results.items():

                print("{}\t{}\n".format(key, val))
        end = timer()
        print("{} processed, {} per second this batch".format((c + 1) * batchsize, int(batchsize / (end - st))),
              end='\r')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model', type=str, default='modelout/models/codegnndualfcinfo_E07.h5')#codegnndualfcinfo_E01.h5
    parser.add_argument('--modeltype', dest='modeltype', type=str, default='codegnndualfcinfo')
    parser.add_argument('--gpu', dest='gpu', type=str, default='')
    parser.add_argument('--data', dest='dataprep', type=str, default='../ICPC2020_GNN/data')
    parser.add_argument('--outdir', dest='outdir', type=str, default='modelout/')
    parser.add_argument('--batch-size', dest='batchsize', type=int, default=1)#128
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
    indexs = ['07']
    for index in indexs:
        write_predict(index)




