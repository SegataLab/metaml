#!/usr/bin/env python

import argparse as ap
import itertools
import numpy as np
import sys
import scipy.stats as stats
from sklearn import metrics

def read_params(args):
	parser = ap.ArgumentParser(description='Compute statistical test on predicted labels for binary classification problems')
	arg = parser.add_argument
	arg( 'inp_f', metavar='INPUT_FILE', nargs='?', default=None, type=str, help="the input file")
	arg( 'inp_f2', metavar='INPUT_FILE2', nargs='?', default=None, type=str, help="the input file2")
	arg( 'out_f', metavar='OUTPUT_FILE', nargs='?', default=None, type=str, help="the output file [stdout if not present]")
	arg( '-p','--runs_cv_folds', default=10, type=int, help="the number of cross-validation folds per run\n")
	return vars(parser.parse_args())

if __name__ == "__main__":
	par = read_params(sys.argv)

	if par['out_f']:
		fidout = open(par['out_f'] + '.txt','w')
	else:
		fidout = sys.stdout

	#read output files
	f = open(par['inp_f']).read().split("#features")[1:]
	f2 = open(par['inp_f2']).read().split("#features")[1:]

	#extract true and estimated labels for each run
	fn = [int(s.split('\t')[1].split('\n')[0]) for s in f]
	fn_r = range(len(fn))
	l_ = [[map(int,t.split('\n')[0].split('\t')[1:-1]) for t in s.split('true labels')[1:]] for s in f]
	l_es_ = [[map(int,t.split('\n')[0].split('\t')[1:-1]) for t in s.split('estimated labels')[1:]] for s in f]
	p_es_pos_ = [[map(float,t.split('\n')[0].split('\t')[1:-1]) for t in s.split('estimated probabilities')[1:]] for s in f]

	f2n = [int(s.split('\t')[1].split('\n')[0]) for s in f2]
	f2n_r = range(len(f2n))
	l2_ = [[map(int,t.split('\n')[0].split('\t')[1:-1]) for t in s.split('true labels')[1:]] for s in f2]
	l2_es_ = [[map(int,t.split('\n')[0].split('\t')[1:-1]) for t in s.split('estimated labels')[1:]] for s in f2]
	p2_es_pos_ = [[map(float,t.split('\n')[0].split('\t')[1:-1]) for t in s.split('estimated probabilities')[1:]] for s in f2]

	#compute metrics for each run/fold
	p = par['runs_cv_folds']
	r = len(l_[0])/p
	i = []
	ii = []
	c = 0
	for j in range(r):
		i.append([j*p+j2 for j2 in range(p) if (len(np.unique(l_[0][j*p+j2]))==2) & (len(np.unique(l2_[0][j*p+j2]))==2)])
		ii.append([j2+c for j2 in range(len(i[-1]))])
		c = c + len(i[-1])
	pr_i = list(itertools.chain(*i))
	pr_ii = list(itertools.chain(*ii))
	
	f_accuracy = [[metrics.accuracy_score(l_[s][j], l_es_[s][j]) for j in pr_i] for s in fn_r]
	f_f1 = [[metrics.f1_score(l_[s][j], l_es_[s][j], pos_label=None, average='weighted') for j in pr_i] for s in fn_r]
	f_precision = [[metrics.precision_score(l_[s][j], l_es_[s][j], pos_label=None, average='weighted') for j in pr_i] for s in fn_r]
	f_recall = [[metrics.recall_score(l_[s][j], l_es_[s][j], pos_label=None, average='weighted') for j in pr_i] for s in fn_r]
	f_auc = [[metrics.roc_auc_score(l_[s][j], p_es_pos_[s][j]) for j in pr_i] for s in fn_r]

	f2_accuracy = [[metrics.accuracy_score(l2_[s][j], l2_es_[s][j]) for j in pr_i] for s in f2n_r]
	f2_f1 = [[metrics.f1_score(l2_[s][j], l2_es_[s][j], pos_label=None, average='weighted') for j in pr_i] for s in f2n_r]
	f2_precision = [[metrics.precision_score(l2_[s][j], l2_es_[s][j], pos_label=None, average='weighted') for j in pr_i] for s in f2n_r]
	f2_recall = [[metrics.recall_score(l2_[s][j], l2_es_[s][j], pos_label=None, average='weighted') for j in pr_i] for s in f2n_r]
	f2_auc = [[metrics.roc_auc_score(l2_[s][j], p2_es_pos_[s][j]) for j in pr_i] for s in f2n_r]

	#compute statistical test
	t_accuracy = [[2*stats.t.sf(np.abs(np.mean([f_accuracy[s][j]-f2_accuracy[s2][j] for j in pr_ii]))/np.mean([np.std([f_accuracy[s][j2]-f2_accuracy[s2][j2] for j2 in ii[j]])/np.sqrt(len(ii[j])) for j in range(r)]),9) for s2 in f2n_r] for s in fn_r]
	t_f1 = [[2*stats.t.sf(np.abs(np.mean([f_f1[s][j]-f2_f1[s2][j] for j in pr_ii]))/np.mean([np.std([f_f1[s][j2]-f2_f1[s2][j2] for j2 in ii[j]])/np.sqrt(len(ii[j])) for j in range(r)]),9) for s2 in f2n_r] for s in fn_r]
	t_precision = [[2*stats.t.sf(np.abs(np.mean([f_precision[s][j]-f2_precision[s2][j] for j in pr_ii]))/np.mean([np.std([f_precision[s][j2]-f2_precision[s2][j2] for j2 in ii[j]])/np.sqrt(len(ii[j])) for j in range(r)]),9) for s2 in f2n_r] for s in fn_r]
	t_recall = [[2*stats.t.sf(np.abs(np.mean([f_recall[s][j]-f2_recall[s2][j] for j in pr_ii]))/np.mean([np.std([f_recall[s][j2]-f2_recall[s2][j2] for j2 in ii[j]])/np.sqrt(len(ii[j])) for j in range(r)]),9) for s2 in f2n_r] for s in fn_r]
	t_auc = [[2*stats.t.sf(np.abs(np.mean([f_auc[s][j]-f2_auc[s2][j] for j in pr_ii]))/np.mean([np.std([f_auc[s][j2]-f2_auc[s2][j2] for j2 in ii[j]])/np.sqrt(len(ii[j])) for j in range(r)]),9) for s2 in f2n_r] for s in fn_r]

	#save statistical test
	for s in range(len(fn)):
		fidout.write('#features for file:\t' + str(fn[s]))
		for s2 in range(len(f2n)):
			fidout.write('\n#features for file2:\t' + str(f2n[s2]))
			fidout.write('\ntest accuracy\t' + str(t_accuracy[s][s2]))
			fidout.write('\ntest f1\t' + str(t_f1[s][s2]))
			fidout.write('\ntest precision\t' + str(t_precision[s][s2]))
			fidout.write('\ntest recall\t' + str(t_recall[s][s2]))
			fidout.write('\ntest auc\t' + str(t_auc[s][s2]))
		fidout.write('\n')

	if par['out_f']:
		fidout.close()