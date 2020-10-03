import keras
import tensorflow as tf
from graphlayer import GCNLayer
from keras.layers import Input, Dense, Embedding, Activation, concatenate, Flatten, CuDNNGRU, TimeDistributed, dot,Bidirectional, LSTM
from keras.models import Model
import argparse
from keras import backend as K
from keras.layers import Lambda

class CodeGNNDualFcInfo:
    def __init__(self, config):
        config['modeltype'] = 'codegnndualfcinfo'
        config['use_pointer'] = True
        config['sdatlen'] = 10
        config['stdatlen'] = 25
        config['tdatlen'] = 50
        config['smllen'] = 100
        config['metlen'] = 17
        config['3dsmls'] = False

        self.config = config
        self.tdatvocabsize = config['tdatvocabsize']
        self.comvocabsize = config['comvocabsize']
        self.smlvocabsize = config['smlvocabsize']
        self.tdatlen = config['tdatlen']
        self.sdatlen = config['sdatlen']
        self.comlen = config['comlen']
        self.smllen = config['maxastnodes']
        self.metlen = config['metlen']
        self.config['batch_maker'] = 'gnn_method'
        self.embdims = 100
        self.smldims = 256
        self.recdims = 256
        self.tdddims = 256

    def create_model(self):
        tdat_input = Input(shape=(self.tdatlen,))
        sdat_input = Input(shape=(self.sdatlen, self.config['stdatlen']))
        com_input = Input(shape=(self.comlen,))
        node_input = Input(shape=(self.smllen,))
        edge_input = Input(shape=(self.smllen, self.smllen))
        met_input = Input(shape=(self.metlen,))
        my_transpose = Lambda(lambda x: K.permute_dimensions(x,(0, 2, 1)))
        edge_input2 = my_transpose(edge_input)

        tdel = Embedding(output_dim=self.embdims, input_dim=self.tdatvocabsize, mask_zero=False)
        tde = tdel(tdat_input)
        se = tdel(node_input)
        me = tdel(met_input)

        tenc = CuDNNGRU(self.recdims, return_state=True, return_sequences=True)
        tencout, tstate_h = tenc(tde)

        menc = CuDNNGRU(self.recdims, return_state=True, return_sequences=True)
        mencout, mstate_h = menc(me)

        de = Embedding(output_dim=self.embdims, input_dim=self.comvocabsize, mask_zero=False)(
            com_input)
        dec = CuDNNGRU(self.recdims, return_sequences=True)
        decout = dec(de, initial_state=tstate_h)

        mattn = dot([decout, mencout], axes=[2, 2])
        mattn = Activation('softmax')(mattn)
        mcontext = dot([mattn, mencout], axes=[2, 1])

        tattn = dot([decout, tencout], axes=[2, 2])
        tattn = Activation('softmax')(tattn)
        tcontext = dot([tattn, tencout], axes=[2, 1])

        semb = TimeDistributed(tdel)
        sde = semb(sdat_input)
        senc = TimeDistributed(CuDNNGRU(int(self.recdims)))
        senc = senc(sde)

        sattn = dot([decout, senc], axes=[2, 2])
        sattn = Activation('softmax')(sattn)
        scontext = dot([sattn, senc], axes=[2, 1])

        astwork1 = se
        astwork2 = se
        for i in range(self.config['asthops']):
            astwork1 = GCNLayer(100)([astwork1, edge_input])
            astwork2 = GCNLayer(100)([astwork2, edge_input2])
        astwork = concatenate([astwork1, astwork2, se])
        astwork = Bidirectional(CuDNNGRU(128, return_sequences=True))(astwork)

        aattn = dot([decout, astwork], axes=[2, 2])
        aattn = Activation('softmax')(aattn)
        acontext = dot([aattn, astwork], axes=[2, 1])

        context = concatenate([scontext,tcontext, decout, acontext, mcontext])

        out = TimeDistributed(Dense(self.tdddims, activation="relu"))(context)
        out = Flatten()(out)
        out = Dense(self.comvocabsize, activation="softmax")(out)

        concate_1 = concatenate([mcontext, decout, de])
        p_gen = TimeDistributed(Dense(1))(concate_1)
        p_gen = Lambda(lambda x: K.mean(x, axis=1))(p_gen)
        p_gen = Activation('sigmoid')(p_gen)
        attn_concate = Lambda(lambda x: K.mean(x, axis=1))(
            mattn)

        if self.config['use_pointer']:
            final_dists = self._calc_final_dist(out, attn_concate, p_gen, met_input)
        else:
            final_dists = out

        model = Model(inputs=[tdat_input,sdat_input,com_input, node_input, edge_input, met_input], outputs=final_dists)
        model.compile(loss='categorical_crossentropy', optimizer='adamax', metrics=['accuracy'])
        return self.config, model

    def _calc_final_dist(self, vocab_dists, attn_dists, p_gen, type_input):

        type_input = K.cast(type_input,tf.int32)
        WeightMultLayer = Lambda(lambda x: x[0] * x[1])
        SupWeightMultLayer = Lambda(lambda x: (1 - x[0]) * x[1])
        DistPlus = Lambda(lambda x: x[0] + x[1])

        vocab_dists = WeightMultLayer([p_gen, vocab_dists])
        attn_dists_weighted = SupWeightMultLayer([p_gen, attn_dists])
        bz = Lambda(lambda x: K.shape(x)[0])(vocab_dists)
        vsize = Lambda(lambda x: K.shape(x)[1])(vocab_dists)
        shape = [bz, vsize]

        def preparation(x):
            batch_nums = tf.range(0, limit=bz)
            batch_nums = K.expand_dims(batch_nums, 1)
            attn_len = K.shape(type_input)[1]
            batch_nums = tf.tile(batch_nums, multiples=[1, attn_len])
            indices = K.stack((batch_nums, type_input), axis=2)
            return indices

        indices = Lambda(preparation)([])
        ScatterNdList = Lambda(
            lambda x: tf.scatter_nd(indices, x, shape=shape, name='making_attn_dists_projected_at_step_0'))
        attn_dists_projected = ScatterNdList(attn_dists_weighted)
        print("attn_dists_projected:{}".format(attn_dists_projected))

        final_dists = DistPlus([vocab_dists, attn_dists_projected])

        def _add_epsilon(epsilon=1e-9):

            _AddEpsilon = Lambda(lambda x: x + K.ones_like(x) * epsilon)
            return _AddEpsilon

        AddEpsilon = _add_epsilon()
        final_dists = AddEpsilon(final_dists)

        return final_dists

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--gpu', type=str, help='0 or 1', default='0')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=200)
    parser.add_argument('--epochs', dest='epochs', type=int, default=10)
    parser.add_argument('--modeltype', dest='modeltype', type=str, default='codegnngru')
    parser.add_argument('--data', dest='dataprep', type=str, default='../data')
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
    # TODO: setup config
    config = dict()
    config['asthops'] = asthops
    config['tdatvocabsize'] = 5000
    config['comvocabsize'] = 5000
    config['smlvocabsize'] = 5000

    config['tdatlen'] = 50
    config['maxastnodes'] = 100
    config['comlen'] = 13

    config['batch_size'] = batch_size
    config['epochs'] = epochs
    mdl = CodeGNNDualFcInfo(config)
    _, model = mdl.create_model()
    print(model.summary())