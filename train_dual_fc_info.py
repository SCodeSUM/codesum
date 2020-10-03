import argparse
import os
import pickle
import random
import sys
import time
import traceback
import numpy as np
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, Callback
import keras.backend as K
from model import create_model
from myutils import batch_gen, init_tf

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--gpu', type=str, help='0 or 1', default='0')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=128)
    parser.add_argument('--epochs', dest='epochs', type=int, default=10)
    parser.add_argument('--modeltype', dest='modeltype', type=str, default='codegnndualfcinfo')
    parser.add_argument('--data', dest='dataprep', type=str, default='../codesum/data')
    parser.add_argument('--outdir', dest='outdir', type=str, default='./modelout')
    parser.add_argument('--asthops', dest='hops', type=int, default=2)
    args = parser.parse_args()

    outdir = args.outdir
    dataprep = args.dataprep
    gpu = args.gpu
    batch_size = args.batch_size
    epochs = args.epochs
    modeltype = args.modeltype
    asthops = args.hops

    init_tf(gpu)

    print('load_tok')
    hightoken = pickle.load(open('{}/hightoken.np'.format(dataprep), 'rb'), encoding='UTF-8')
    tdatstok = pickle.load(open('{}/tdatstok.pkl'.format(dataprep), 'rb'), encoding='UTF-8')
    asttok = pickle.load(open('{}/smls.tok'.format(dataprep), 'rb'), encoding='UTF-8')

    tdatvocabsize = tdatstok.vocab_size
    comvocabsize = tdatstok.vocab_size
    astvocabsize = tdatstok.vocab_size

    # TODO: setup config
    config = dict()
    config['asthops'] = asthops
    config['tdatvocabsize'] = 5000
    config['comvocabsize'] = 5000
    config['smlvocabsize'] = 5000

    config['tdatlen'] = 50
    config['maxastnodes'] = 100
    config['comlen'] = 13
    config['metlen'] = 17

    config['batch_size'] = batch_size
    config['epochs'] = epochs

    print('load dataset.pkl')
    seqdata = pickle.load(open('{}/dataset.pkl'.format(dataprep), 'rb'))
    node_data = seqdata['strain_nodes']
    edges = seqdata['strain_edges']
    met_train = pickle.load(open('{}/mtrain.pkl'.format(dataprep), 'rb'))
    met_val = pickle.load(open('{}/mval.pkl'.format(dataprep), 'rb'))
    met_test = pickle.load(open('{}/mtest.pkl'.format(dataprep), 'rb'))
    com_train = pickle.load(open('{}/com_train.pkl'.format(dataprep), 'rb'))
    com_val = pickle.load(open('{}/com_val.pkl'.format(dataprep), 'rb'))
    com_test = pickle.load(open('{}/com_test.pkl'.format(dataprep), 'rb'))
    config['edge_type'] = 'sml'

    steps = int(len(seqdata['ctrain']) / batch_size) + 1
    valsteps = int(len(seqdata['cval']) / batch_size) + 1

    print('tdatvocabsize {}'.format(tdatvocabsize))
    print('comvocabsize {}'.format(comvocabsize))
    print('smlvocabsize {}'.format(astvocabsize))
    print('batch size {}'.format(batch_size))
    print('steps {}'.format(steps))
    print('training data size {}'.format(steps * batch_size))
    print('vaidation data size {}'.format(valsteps * batch_size))
    print('------------------------------------------')

    config, model = create_model(modeltype, config)
    print(model.summary())

    gen = batch_gen(seqdata, 'train', config, nodedata=node_data, edgedata=edges,metdata=met_train,comdata=com_train,hightoken=hightoken)
    checkpoint = ModelCheckpoint(outdir + "/models/" + modeltype + "_E{epoch:02d}.h5",save_weights_only=True)

    valgen = batch_gen(seqdata, 'val', config, nodedata=seqdata['sval_nodes'], edgedata=seqdata['sval_edges'],metdata=met_val,comdata=com_val,hightoken=hightoken)
    callbacks = [checkpoint]

    model.fit_generator(gen, steps_per_epoch=steps, epochs=epochs, verbose=1, max_queue_size=4,
                        callbacks=callbacks, validation_data=valgen, validation_steps=valsteps)
