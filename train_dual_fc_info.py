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
from utils.model import create_model
from utils.myutils import batch_gen, init_tf
from keras.utils import multi_gpu_model
from keras.models import load_model
from codesum.models.custom.graphlayer import GCNLayer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--gpu', type=str, help='0 or 1', default='0')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=128)
    parser.add_argument('--epochs', dest='epochs', type=int, default=10)
    parser.add_argument('--modeltype', dest='modeltype', type=str, default='codegnndualfcinfo')
    parser.add_argument('--data', dest='dataprep', type=str, default='./new_data')
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

    # set gpu here
    init_tf(gpu)

    # Load tokenizers
    print('load_tok')
    with open('{}/hightoken.np'.format(dataprep), 'rb') as f:
        hightoken = pickle.load(f)
    with open('{}/tdatstok.pkl'.format(dataprep), 'rb') as f:
        tdatstok = pickle.load(f)
    with open('{}/smls.tok'.format(dataprep), 'rb') as f:
        asttok = pickle.load(f)

    tdatvocabsize = tdatstok.vocab_size
    comvocabsize = tdatstok.vocab_size
    astvocabsize = tdatstok.vocab_size

    # setup config
    config = dict()
    config['asthops'] = asthops

    # dynamic vocabulary
    config['tdatvocabsize'] = 5000 #tdatvocabsize
    config['comvocabsize'] = 5000 #comvocabsize
    config['smlvocabsize'] = 5000 #astvocabsize

    # all vocabulary 75,437
    config['tdatvocabsize'] = 75437 #tdatvocabsize
    config['comvocabsize'] = 75437 #comvocabsize
    config['smlvocabsize'] = 75437 #astvocabsize

    # set sequence length for our input
    config['tdatlen'] = 50
    config['maxastnodes'] = 100
    config['comlen'] = 13
    config['metlen'] = 17

    config['batch_size'] = batch_size
    config['epochs'] = epochs

    # Load data
    print('load dataset.pkl')
    with open('{}/dataset.pkl'.format(dataprep), 'rb') as f:
        seqdata = pickle.load(f)

    node_data = seqdata['strain_nodes']
    edges = seqdata['strain_edges']
    with open('{}/mtrain.pkl'.format(dataprep), 'rb') as f:
        met_train = pickle.load(f)
    with open('{}/mval.pkl'.format(dataprep), 'rb') as f:
        met_val = pickle.load(f)
    with open('{}/mtest.pkl'.format(dataprep), 'rb') as f:
        met_test = pickle.load(f)
    with open('{}/com_train.pkl'.format(dataprep), 'rb') as f:
        com_train = pickle.load(f)
    with open('{}/com_val.pkl'.format(dataprep), 'rb') as f:
        com_val = pickle.load(f)
    with open('{}/com_test.pkl'.format(dataprep), 'rb') as f:
        com_test = pickle.load(f)
    config['edge_type'] = 'sml'

    # setup parameters
    steps = int(len(seqdata['ctrain']) / batch_size) + 1
    valsteps = int(len(seqdata['cval']) / batch_size) + 1

    # Print information
    print('tdatvocabsize {}'.format(tdatvocabsize))
    print('comvocabsize {}'.format(comvocabsize))
    print('smlvocabsize {}'.format(astvocabsize))
    print('batch size {}'.format(batch_size))
    print('steps {}'.format(steps))
    print('training data size {}'.format(steps * batch_size))

    print('vaidation data size {}'.format(valsteps * batch_size))
    print('------------------------------------------')

    # create model
    print(config)

    config, model = create_model(modeltype, config)
    print(model.summary())

    # set up data generators
    gen = batch_gen(seqdata, 'train', config, nodedata=node_data, edgedata=edges,metdata=met_train,comdata=com_train,hightoken=hightoken)

    checkpoint = ModelCheckpoint(outdir + "/models/" + modeltype + "_vfull_E{epoch:02d}.h5",save_weights_only=True)

    valgen = batch_gen(seqdata, 'val', config, nodedata=seqdata['sval_nodes'], edgedata=seqdata['sval_edges'],metdata=met_val,comdata=com_val,hightoken=hightoken)
    callbacks = [checkpoint]

    model.fit_generator(gen, steps_per_epoch=steps, epochs=epochs, verbose=2, max_queue_size=4,
                        callbacks=callbacks, validation_data=valgen, validation_steps=valsteps)
