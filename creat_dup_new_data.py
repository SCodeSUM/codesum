import pickle
import json

# Remove code clones, creat new improved dataset
with open('data/dataset.pkl', 'rb') as f:
    seqdata = pickle.load(f)
with open('data/com_train.pkl', 'rb') as f:
    com_train = pickle.load(f)
with open('data/com_val.pkl', 'rb') as f:
    com_val = pickle.load(f)
with open('data/com_test.pkl', 'rb') as f:
    com_test = pickle.load(f)
with open('data/mtrain.pkl', 'rb') as f:
    mtrain = pickle.load(f)
with open('data/mval.pkl', 'rb') as f:
    mval = pickle.load(f)
with open('data/mtest.pkl', 'rb') as f:
    mtest = pickle.load(f)

train_id = list(seqdata['dttrain'].keys())
val_id = list(seqdata['dtval'].keys())
test_id = list(seqdata['dttest'].keys())

print(seqdata['config'])

print('old dataset train length',len(train_id))
print('old dataset val length',len(val_id))
print('old dataset test length',len(test_id))
print(train_id[:10])
with open('dup_data_result/DuplicateCodeDetector.csproj.json', 'r') as file:
    dup_code = json.load(file)

flatlist = []
for lst in dup_code:
    flatlist += [int(d) for d in lst[1:]]
print('Number of duplicate files filtered',len(flatlist))


dup_train_id = list(set(train_id)&set(flatlist))
print('train: Number of duplicate files',len(dup_train_id))
dup_val_id = list(set(val_id)&set(flatlist))
print('val: Number of duplicate files',len(dup_val_id))
dup_test_id = list(set(test_id)&set(flatlist))
print('test: Number of duplicate files',len(dup_test_id))


[seqdata['ctrain'].pop(k) for k in dup_train_id]
[seqdata['cval'].pop(k) for k in dup_val_id]
[seqdata['ctest'].pop(k) for k in dup_test_id]
[seqdata['dstrain'].pop(k) for k in dup_train_id]
[seqdata['dsval'].pop(k) for k in dup_val_id]
[seqdata['dstest'].pop(k) for k in dup_test_id]
[seqdata['dttrain'].pop(k) for k in dup_train_id]
[seqdata['dtval'].pop(k) for k in dup_val_id]
[seqdata['dttest'].pop(k) for k in dup_test_id]
[seqdata['strain_nodes'].pop(k) for k in dup_train_id]
[seqdata['sval_nodes'].pop(k) for k in dup_val_id]
[seqdata['stest_nodes'].pop(k) for k in dup_test_id]
[seqdata['strain_edges'].pop(k) for k in dup_train_id]
[seqdata['sval_edges'].pop(k) for k in dup_val_id]
[seqdata['stest_edges'].pop(k) for k in dup_test_id]

[com_train.pop(k) for k in dup_train_id]
[com_val.pop(k) for k in dup_val_id]
[com_test.pop(k) for k in dup_test_id]
[mtrain.pop(k) for k in dup_train_id]
[mval.pop(k) for k in dup_val_id]
[mtest.pop(k) for k in dup_test_id]

print(len(list(seqdata['ctrain'].keys())))
print(len(list(seqdata['cval'].keys())))
print(len(list(seqdata['ctest'].keys())))
print(len(list(seqdata['dstrain'].keys())))
print(len(list(seqdata['dsval'].keys())))
print(len(list(seqdata['dstest'].keys())))
print(len(list(seqdata['dttrain'].keys())))
print(len(list(seqdata['dtval'].keys())))
print(len(list(seqdata['dttest'].keys())))
print(len(list(seqdata['strain_nodes'].keys())))
print(len(list(seqdata['sval_nodes'].keys())))
print(len(list(seqdata['stest_nodes'].keys())))
print(len(list(seqdata['strain_edges'].keys())))
print(len(list(seqdata['sval_edges'].keys())))
print(len(list(seqdata['stest_edges'].keys())))

with open('new_data/dup_dataset.pkl', 'wb') as f:
    pickle.dump(seqdata, f)

with open('new_data/com_train.pkl', 'wb') as f:
    pickle.dump(com_train, f)
with open('new_data/com_val.pkl', 'wb') as f:
    pickle.dump(com_val, f)
with open('new_data/com_test.pkl', 'wb') as f:
    pickle.dump(com_test, f)
with open('new_data/mtrain.pkl', 'wb') as f:
    pickle.dump(mtrain, f)
with open('new_data/mval.pkl', 'wb') as f:
    pickle.dump(mval, f)
with open('new_data/mtest.pkl', 'wb') as f:
    pickle.dump(mtest, f)