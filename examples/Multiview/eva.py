import sys
import os
import numpy as np
import cPickle

def compute_label(probs):
    scores = np.array(probs);
    mean_score = np.mean(scores, axis = 0);
    return np.where(mean_score == mean_score.max());

def compute_mean(probs):
    scores = np.array(probs);
    return np.mean(scores, axis = 0);

def parse_res(infile):
    f=open(infile);
    ret = [];
    lines = [i.strip() for i in f];
    f.close();
    index = 0;
    while index<len(lines):
        imagefile = lines[index].split()[0];
        num = int(lines[index].split()[1]);
        dim = int(lines[index].split()[2]);
        index += 1;
        img_id = int(imagefile.split('/')[-1].split('.')[0].split('_')[-1]);
        probs=[];
        for i in range(num):
            prob = [];
            scores = lines[index].split();
            index += 1;
            if not (len(scores) == dim):
                print "error probs";
                sys.exit(1);
            for p in scores:
                prob.append(float(p));
            probs.append(prob);
        label=compute_mean(probs);
        ret.append((img_id, label));


    return ret;

def parse_gt(infile):
    fin = open(infile);
    ret = [];
    for item in fin:
        ret.append(int(item.strip().split()[-1])+1);

    fin.close();
    return ret;

def compute_acc(gt, res, top):
    acc = 0;
    if not len(gt) == len(res):
        print "res is less";
        sys.exit(1);
    res = sorted(res, key=lambda x:x[0]);
    for i in range(len(gt)):
        label = gt[i];
        prob = list(res[i][-1]);
        prob = [(i+1,prob[i]) for i in range(len(prob))];
        prob = sorted(prob, key=lambda x:x[1], reverse = True);
        prob_labels = prob[0:top];
        prob_labels = [i[0] for i in prob_labels];
        if label in prob_labels:
            acc += 1;
    print acc, len(gt), float(acc)/len(gt);

def main(gtfile, resfile):
    gt = parse_gt(gtfile);
    res = parse_res(resfile);
    """
    if os.path.exists('tmp_res'):
        res = cPickle.load(open('tmp_res'));
    else:
        res = parse_res(resfile);
        fo=open('tmp_res', 'w');
        cPickle.dump(res, fo);
        fo.close();
        """
    compute_acc(gt, res, 5);

if __name__=="__main__":
    if len(sys.argv) < 3:
        print "Usage:%s groundtruth result"%(sys.argv[0]);
        sys.exit(1);

    main(sys.argv[1], sys.argv[2]);

        
