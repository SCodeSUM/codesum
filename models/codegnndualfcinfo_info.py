import keras
import keras.utils
import tensorflow as tf
from codesum.models.custom.graphlayer import GCNLayer
from keras.layers import Input, Dense, Embedding, Activation, concatenate, Flatten, CuDNNGRU, TimeDistributed, dot,Bidirectional, LSTM
from keras.models import Model
import argparse
from keras import backend as K
from keras.layers import Lambda

class CodeGNNDualFcInfo_Info:
    def __init__(self, config):
        # init params
        config['modeltype'] = 'codegnndualfcinfo_info'
        config['use_pointer'] = True  # pointer
        config['sdatlen'] = 10  # num of function
        config['stdatlen'] = 25  # num of token per function
        config['tdatlen'] = 50  # num of tokens in the code/text sequence
        config['smllen'] = 100  # max tokens in the ﬂattened AST sequence
        config['metlen'] = 17  # num of method name token
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
        tdat_input = Input(shape=(self.tdatlen,))  # (None, 50)input_1
        sdat_input = Input(shape=(self.sdatlen, self.config['stdatlen']))   #(None,10,25)input_2 context
        com_input = Input(shape=(self.comlen,))  # (None, 13)input_3
        node_input = Input(shape=(self.smllen,))  # (None, 100)input_4
        edge_input = Input(shape=(self.smllen, self.smllen))  # (None, 100, 100)input_5
        met_input = Input(shape=(self.metlen,))  # (None,17)input_6
        my_transpose = Lambda(lambda x: K.permute_dimensions(x,(0, 2, 1)))
        edge_input2 = my_transpose(edge_input) # (None, 100, 100)lambda_1_1

        tdel = Embedding(output_dim=self.embdims, input_dim=self.tdatvocabsize, mask_zero=False)  # 100*10000 embedding_1
        tde = tdel(tdat_input)  # (None, 50, 100)embedding_1[0][0]

        se = tdel(node_input)  # (None, 100, 100)embedding_1[1][0]

        tenc = CuDNNGRU(self.recdims, return_state=True, return_sequences=True)#cu_dnngru_1
        tencout, tstate_h = tenc(tde)  # (None, 50, 256), (N 274944


        de = Embedding(output_dim=self.embdims, input_dim=self.comvocabsize, mask_zero=False)(
            com_input)  # 100*10000 (None, 13, 100)embedding_2
        dec = CuDNNGRU(self.recdims, return_sequences=True) #cu_dnngru_3
        decout = dec(de, initial_state=tstate_h)  # (None, 13, 256)


        tattn = dot([decout, tencout], axes=[2, 2])  # (None, 13, 50) dot_3
        tattn = Activation('softmax')(tattn) #activation_2
        tcontext = dot([tattn, tencout], axes=[2, 1])  # (None, 13, 256) dot_4


        semb = TimeDistributed(tdel) #time_distributed_1
        sde = semb(sdat_input)

        # sdats encoder
        senc = TimeDistributed(CuDNNGRU(int(self.recdims))) #time_distributed_2
        senc = senc(sde)

        # attention to sdats
        sattn = dot([decout, senc], axes=[2, 2]) #(None, 13, 10)dot_5
        sattn = Activation('softmax')(sattn) #activation_3
        scontext = dot([sattn, senc], axes=[2, 1])

        top2down = se
        bottom2up = se
        for i in range(self.config['asthops']):
            top2down = GCNLayer(100)(
                [top2down, edge_input])  # 1:(None, 100, 100) 2:(None, 100, 100) gcn_layer_1->gcn_layer_3
            bottom2up = GCNLayer(100)([bottom2up, edge_input2])  # gcn_layer_2->gcn_layer_4

        dualGraph = concatenate([top2down, bottom2up, se])  # concatenate_1
        dualGraph = Bidirectional(CuDNNGRU(128, return_sequences=True))(dualGraph)  # bidirectional_1

        aattn = dot([decout, dualGraph], axes=[2, 2])  # (None, 13, 100) dot_7 (Dot)
        aattn = Activation('softmax')(aattn)  # activation_4
        acontext = dot([aattn, dualGraph], axes=[2, 1])  # (None, 13, 256)

        context = concatenate([scontext, tcontext, decout, acontext])  # (None, 13, 768)

        out = TimeDistributed(Dense(self.tdddims, activation="relu"))(context)  # (None, 13, 256)
        out = Flatten()(out)  # (None, 3328)
        out = Dense(self.comvocabsize, activation="softmax")(out)  # (None, 10000)

        if self.config['use_pointer']:
            final_dists = out
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

        vocab_dists = WeightMultLayer([p_gen, vocab_dists]) #(None, 1)*(None, 5000)=>(None, 5000)
        attn_dists_weighted = SupWeightMultLayer([p_gen, attn_dists]) #(1-p_gen)*attn_dists(None, 1)，(None, 50)=>(None, 50)

        bz = Lambda(lambda x: K.shape(x)[0])(vocab_dists)  # hidden_dim batch
        vsize = Lambda(lambda x: K.shape(x)[1])(vocab_dists)

        shape = [bz, vsize]  # (None,5000)

        def preparation(x):
            batch_nums = tf.range(0, limit=bz)  # shape (batch_size)
            batch_nums = K.expand_dims(batch_nums, 1)  # shape (batch_size, 1)
            attn_len = K.shape(type_input)[1]  # number of states we attend over(None, 50)->50
            batch_nums = tf.tile(batch_nums, multiples=[1, attn_len]) #(None,50)
            indices = K.stack((batch_nums, type_input), axis=2) #(None,50,None)
            return indices

        indices = Lambda(preparation)([])
        ScatterNdList = Lambda(
            lambda x: tf.scatter_nd(indices, x, shape=shape, name='making_attn_dists_projected_at_step_0'))
        attn_dists_projected = ScatterNdList(attn_dists_weighted) #(None, 50)->(None,5000+o)
        print("attn_dists_projected:{}".format(attn_dists_projected))

        final_dists = DistPlus([vocab_dists, attn_dists_projected]) #(None,5000+o)->(None,5000+o)

        def _add_epsilon(epsilon=1e-9):
            # return add-epsilon layer
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
    mdl = CodeGNNDualFcInfo_Info(config)
    _, model = mdl.create_model()
    print(model.summary())