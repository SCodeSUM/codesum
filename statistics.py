import jsonlines
import pickle

# statistics

fi_train = open('./compute_dup_data/dats.train', 'r')
func_train_list = fi_train.readlines()
fi_train.close()

fi_valid = open('./compute_dup_data/dats.val', 'r')
func_valid_list = fi_valid .readlines()
fi_valid.close()

fi_test = open('./compute_dup_data/dats.test', 'r')
func_test_list = fi_test.readlines()
fi_test.close()

print(len(func_train_list))
print(len(func_valid_list))
print(len(func_test_list))
print(len(func_train_list)+len(func_valid_list)+len(func_test_list))

seqdata = pickle.load(open('{}/dataset.pkl'.format("../codesum/data"), 'rb'))
print(seqdata.keys())
train_id = seqdata['dttrain'].keys()
val_id = seqdata['dtval'].keys()
test_id = seqdata['dttest'].keys()
print(len(train_id))
print(len(val_id))
print(len(test_id))
print(len(set(list(train_id)+list(val_id)+list(test_id))))
