import rouge

def prepare_results(p, r, f, metric):
    return '\t{}:\t{}: {:5.2f}\t{}: {:5.2f}\t{}: {:5.2f}'.format(metric, 'P', 100.0 * p, 'R', 100.0 * r, 'F1',
                                                                 100.0 * f)
def rouge_score(hyps, refs, ne):
    evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l', 'rouge-w'],
                            max_n=4,
                            limit_length=True,
                            length_limit=100,
                            length_limit_type='words',
                            alpha=.5,  # Default F1_score
                            weight_factor=1.2)
    scores = evaluator.get_scores(hyps, refs)
    return scores

def get_rouge_score(hyps, refs, ne):
    scores = rouge_score(hyps, refs, ne)
    return scores

def print_scores(scores):
    for metric, results in sorted(scores.items(), key=lambda x: x[0]):
        print(prepare_results(results['p'], results['r'], results['f'], metric))

def print_indiv_scores(scores):
    print("Max:")
    print_scores(scores[0])
    print("Min:")
    print_scores(scores[1])
    print("Mean:")
    print_scores(scores[2])
    print("Median:")
    print_scores(scores[3])
    print("Stdev:")
    print_scores(scores[4])

def fil(com):
    ret = list()
    for w in com:
        if not '<' in w:
            ret.append(w)
    return ret

def compute_rouge(input_file, target_file):
    delim = '\t'

    if input_file is None:
        print('no input file')
        exit()

    if target_file is None:
        print('no target file')

    preds = dict()
    predicts = open(input_file, 'r')
    for c, line in enumerate(predicts):
        (fid, pred) = line.split(delim)
        fid = int(fid)
        pred = pred.split()
        pred = fil(pred)
        preds[fid] = ' '.join(pred)
    predicts.close()

    refs = list()
    newpreds = list()

    targets = open(target_file, 'r')
    for line in targets:
        (fid, com) = line.split(',')
        fid = int(fid)
        com = com.split()
        com = ' '.join(fil(com))
        if len(com) < 1:
            continue
        try:
            newpreds.append(preds[fid])
        except Exception as ex:
            continue
        refs.append(com)
    targets.close()

    print_scores(get_rouge_score(newpreds, refs, False))

if __name__ == '__main__':

    # input_file is the file about generated summaries
    # target_file is the file about humman summaries
    input_file = '../modelout/predictions/predict-codegnndualfcinfo_v1000_E01.txt'
    target_file = '../reference/coms.test'
    compute_rouge(input_file,target_file)