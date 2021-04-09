import re
from nltk.translate.bleu_score import corpus_bleu

def fil(com):
    ret = list()
    for w in com:
        if not '<' in w:
            ret.append(w)
    return ret

def bleu_so_far(refs, preds):
    Ba = corpus_bleu(refs, preds)
    B1 = corpus_bleu(refs, preds, weights=(1, 0, 0, 0))
    B2 = corpus_bleu(refs, preds, weights=(0, 1, 0, 0))
    B3 = corpus_bleu(refs, preds, weights=(0, 0, 1, 0))
    B4 = corpus_bleu(refs, preds, weights=(0, 0, 0, 1))

    Ba = round(Ba * 100, 2)
    B1 = round(B1 * 100, 2)
    B2 = round(B2 * 100, 2)
    B3 = round(B3 * 100, 2)
    B4 = round(B4 * 100, 2)

    ret = ''
    ret += ('for %s functions\n' % (len(preds)))
    ret += ('Ba %s\n' % (Ba))
    ret += ('B1 %s\n' % (B1))
    ret += ('B2 %s\n' % (B2))
    ret += ('B3 %s\n' % (B3))
    ret += ('B4 %s\n' % (B4))

    return ret

def compute_bleu(input_file,target_file):
    delim = '\t'

    if input_file is None:
        print('Please provide an input file to test with --input')
        exit()

    preds = dict()
    predicts = open(input_file, 'r')
    for c, line in enumerate(predicts):
        (fid, pred) = line.split(delim)
        fid = int(fid)
        pred = pred.split()
        pred = fil(pred)
        preds[fid] = pred
    predicts.close()


    refs = list()
    newpreds = list()
    predsindex = list()

    targets =  open(target_file, 'r')
    for line in targets:
        (fid, com) = line.split(',')
        fid = int(fid)
        com = com.split()
        com = fil(com)
        try:
            newpreds.append(preds[fid])
            predsindex.append(fid)
        except KeyError as ex:
            continue
        refs.append([com])
    targets.close()

    print(bleu_so_far(refs, newpreds))

if __name__ == '__main__':
    # input_file is the file about generated summaries
    # target_file is the file about humman summaries
    input_file = '../modelout/predictions/predict-codegnndualfcinfo_v1000_E01.txt'
    target_file = '../reference/coms.test'

    compute_bleu(input_file, target_file)
