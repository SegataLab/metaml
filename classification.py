#!/usr/bin/env python

import argparse as ap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from sklearn import decomposition
from sklearn import metrics
from sklearn import preprocessing as prep
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNetCV
from sklearn.svm import SVC

class class_metrics:
	def __init__(self):
		self.accuracy = []
		self.f1 = []
		self.precision = []
		self.recall = []
		self.auc = []
		self.roc_curve = []
		self.confusion_matrix = []

class class_params:
	def __init__(self):
		self.learner_type = []
		self.feature_selection = []
		self.cv_folds = []
		self.cv_grid = []
		self.cv_scoring = []
		self.fs_grid = []

class feature_importance:
	def __init__(self, feat, p):
		self.feat_sel = feat
		self.imp = np.array([p]*len(feat))

def compute_feature_importance(el, feat, feat_sel, ltype):
	fi = feature_importance(feat, 0.0)
	if ltype == 'rf':
		t = el.feature_importances_
	elif (ltype == 'lasso') | (ltype == 'enet'):
		t = abs(el.coef_)/sum(abs(el.coef_))
	else:
		t = [1.0/len(feat_sel)]*len(feat_sel)
	ti = [feat.index(s) for s in feat_sel]
	fi.imp[ti] = t

	t = sorted(range(len(t)), key=lambda s: t[s], reverse=True)
	fi.feat_sel = [feat[ti[s]] for s in t if fi.imp[ti[s]] != 0]

	return fi

def plot_pca(f, l, feat_sel):
	f = f[feat_sel].values	
	l = l.values.flatten().astype('int')
	
	pca = decomposition.PCA(n_components=2)
	pca = pca.fit(f)
	ft = pca.transform(f)

	fig, ax = plt.subplots()
	for i in np.unique(l):
		ax.scatter(ft[l==i, 0], ft[l==i, 1], c=l[l==i], s=200, cmap=plt.cm.jet, vmin=min(l), vmax=max(l), edgecolors='None', alpha=0.6)
		ax.text(ft[l==i, 0].mean(), ft[l==i, 1].mean(), 'Class ' + str(i), horizontalalignment='center', bbox=dict(alpha=0.5, edgecolor='w', facecolor='w'))

	ax.set_xlabel('PC1 (' + str(np.round(100*pca.explained_variance_ratio_[0], decimals=2)) + '%)')
	ax.set_ylabel('PC2 (' + str(np.round(100*pca.explained_variance_ratio_[1], decimals=2)) + '%)')
	ax.set_xlim([min(ft[:, 0]),max(ft[:, 0])])
	ax.set_ylim([min(ft[:, 1]),max(ft[:, 1])])	
	
	fig.savefig(par['out_f'] + '_pca.' + par['figure_extension'], bbox_inches='tight')

def read_params(args):
	parser = ap.ArgumentParser(description='MetAML - Metagenomic prediction Analysis based on Machine Learning')
	arg = parser.add_argument
	arg( 'inp_f', metavar='INPUT_FILE', nargs='?', default=sys.stdin, type=str, help="the input dataset file [stdin if not present]")
	arg( 'out_f', metavar='OUTPUT_FILE', nargs='?', default=None, type=str, help="the output file [stdout if not present]")

	arg( '-z','--feature_identifier', default='k__', type=str, help="the feature identifier\n")

	arg( '-d','--define', type=str, help="define the classification problem\n")
	arg( '-t','--target', type=str, help="define the target domain\n")
	arg( '-u','--unique', type=str, help="the unique samples to select\n")
	arg( '-b','--label_shuffling', action='store_true', help="label shuffling\n")

	arg( '-r','--runs_n', default=20, type=int, help="the number of runs\n")
	arg( '-p','--runs_cv_folds', default=10, type=int, help="the number of cross-validation folds per run\n")
	arg( '-w','--set_seed', action='store_true', help="setting seed\n")

	arg( '-l','--learner_type', choices=['rf','svm','lasso','enet'], help="the type of learner/classifier\n")
	arg( '-i','--feature_selection', choices=['lasso','enet'], help="the type of feature selection\n")
	arg( '-f','--cv_folds', type=int, help="the number of cross-validation folds for model selection\n")
	arg( '-g','--cv_grid', type=str, help="the parameter grid for model selection\n")
	arg( '-s','--cv_scoring', type=str, help="the scoring function for model selection\n")
	arg( '-j','--fs_grid', type=str, help="the parameter grid for feature selection\n")

	arg( '-e','--figure_extension', default='png', type=str, help="the extension of output figure\n")
	
	return vars(parser.parse_args())

def save_average_feature_importance(fi, feat):
	fi_ave = feature_importance(feat, 0.0)

	t = [s.imp for s in fi]
	fi_ave_std = np.std(t, axis=0)
	t = np.mean(t, axis=0)
	fi_ave.imp = t
	t = sorted(range(len(t)), key=lambda s: t[s], reverse=True)
	fi_ave.feat_sel = [feat[s] for s in t if fi_ave.imp[s] != 0]

	fidout.write('Feature importance (ranking, name, average, std)\n')
	[fidout.write(str(s) + '\t' + feat[t[s]] + '\t' + str(fi_ave.imp[t[s]]) + '\t' + str(fi_ave_std[t[s]]) + '\n') for s in range(len(t))]

	return fi_ave

def save_results(l, l_es, p_es, i_tr, i_u, nf, runs_n, runs_cv_folds):
	n_clts = len(np.unique(l))
	cm = class_metrics()

	if par['out_f']:
		fidoutes.write('#features\t' + str(nf) + '\n')
		if n_clts == 2:
			fidoutroc.write('#features\t' + str(nf) + '\n')	

	for j in range(runs_n*runs_cv_folds):
		l_ = pd.DataFrame([l.loc[i] for i in l[-i_tr[j] & i_u[j/runs_cv_folds]].index]).values.flatten().astype('int')
		l_es_ = l_es[j].values.flatten().astype('int')
		if (lp.learner_type == 'rf') | (lp.learner_type == 'svm'):
			p_es_pos_ = p_es[j].loc[:,1].values
		else:
			p_es_pos_ = p_es[j].loc[:,0].values
		ii_ts_ = [i for i in range(len(i_tr[j])) if i_tr[j][i]==False]

		cm.accuracy.append(metrics.accuracy_score(l_, l_es_))
		cm.f1.append(metrics.f1_score(l_, l_es_, pos_label=None, average='weighted'))
		cm.precision.append(metrics.precision_score(l_, l_es_, pos_label=None, average='weighted'))
		cm.recall.append(metrics.recall_score(l_, l_es_, pos_label=None, average='weighted'))
		if len(np.unique(l_)) == n_clts:
			if n_clts == 2:
				cm.auc.append(metrics.roc_auc_score(l_, p_es_pos_))
				cm.roc_curve.append(metrics.roc_curve(l_, p_es_pos_))
				fidoutroc.write('run/fold\t' + str(j/runs_cv_folds) + '/' + str(j%runs_cv_folds) + '\n')
				for i in range(len(cm.roc_curve[-1])):
					for i2 in range(len(cm.roc_curve[-1][i])):
						fidoutroc.write(str(cm.roc_curve[-1][i][i2]) + '\t')
					fidoutroc.write('\n')
			cm.confusion_matrix.append(metrics.confusion_matrix(l_, l_es_, labels=np.unique(l.astype('int'))))

		if par['out_f']:
			fidoutes.write('run/fold\t' + str(j/runs_cv_folds) + '/' + str(j%runs_cv_folds))
			fidoutes.write('\ntrue labels\t')
			[fidoutes.write(str(i)+'\t') for i in l_]
			fidoutes.write('\nestimated labels\t')
			[fidoutes.write(str(i)+'\t') for i in l_es_]
			if n_clts <= 2:
				fidoutes.write('\nestimated probabilities\t')
				[fidoutes.write(str(i)+'\t') for i in p_es_pos_]
			fidoutes.write('\nsample index\t')			
			[fidoutes.write(str(i)+'\t') for i in ii_ts_]
			fidoutes.write('\n')

	fidout.write('#samples\t' + str(sum(sum(i_u))/len(i_u)))
	fidout.write('\n#features\t' + str(nf))
	fidout.write('\n#runs\t' + str(runs_n))
	fidout.write('\n#runs_cv_folds\t' + str(runs_cv_folds))	

	fidout.write('\naccuracy\t' + str(np.mean(cm.accuracy)) + '\t' + str(np.std(cm.accuracy)))
	fidout.write('\nf1\t' + str(np.mean(cm.f1)) + '\t' + str(np.std(cm.f1)))
	fidout.write('\nprecision\t' + str(np.mean(cm.precision)) + '\t' + str(np.std(cm.precision)))
	fidout.write('\nrecall\t' + str(np.mean(cm.recall)) + '\t' + str(np.std(cm.recall)))
	if n_clts == 2:
		fidout.write('\nauc\t' + str(np.mean(cm.auc)) + '\t' + str(np.std(cm.auc)))
	else:
		fidout.write('\nauc\t[]\t[]')
	fidout.write('\nconfusion matrix')
	if len(cm.confusion_matrix) > 0:
		for i in range(len(cm.confusion_matrix[0])):
			for i2 in range(len(cm.confusion_matrix[0][i])):
				fidout.write('\t' + str(np.sum([cm.confusion_matrix[j][i][i2] for j in range(len(cm.confusion_matrix))])))
			fidout.write('\n')
	else:
		fidout.write('\n')

	return cm

def set_class_params(args, l):
	lp = class_params()

	if par['learner_type']:
		lp.learner_type = par['learner_type']
		if (max(l.values.flatten().astype('int'))>1) & (lp.learner_type != 'svm'):
			lp.learner_type = 'rf'	
	else:
		lp.learner_type = 'rf'

	if par ['feature_selection']:
		lp.feature_selection = par['feature_selection']
	else:
		lp.feature_selection = 'none'

	if par['cv_folds']:
		lp.cv_folds = int(par['cv_folds'])
	else:
		lp.cv_folds = 5

	if par['cv_grid']:
		lp.cv_grid = eval(par['cv_grid'])
	elif lp.learner_type == 'rf':
		lp.cv_grid = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 125, 150, 175, 200]
	elif lp.learner_type == 'svm':
		if par['feature_identifier'] == 'k__':
			# lp.cv_grid = [ {'C': [2**s for s in range(-5,16,2)], 'kernel': ['linear']}, {'C': [2**s for s in range(-5,16,2)], 'gamma': [2**s for s in range(3,-15,-2)], 'kernel': ['rbf']} ]
			lp.cv_grid = [ {'C': [1, 10, 100, 1000], 'kernel': ['linear']}, {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']} ]
		else:
			lp.cv_grid = [ {'C': [1, 10, 100, 1000], 'kernel': ['linear']}]
	elif lp.learner_type == 'lasso':
		lp.cv_grid = [np.logspace(-4, -0.5, 50)]
	elif lp.learner_type == 'enet':
		lp.cv_grid = [np.logspace(-4, -0.5, 50), [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]]

	if par['fs_grid']:
		lp.fs_grid = eval(par['fs_grid'])
	elif lp.feature_selection == 'lasso':
		lp.fs_grid = [np.logspace(-4, -0.5, 50)]
	elif lp.feature_selection == 'enet':
		lp.fs_grid = [np.logspace(-4, -0.5, 50), [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]]
	
	if par['cv_scoring']:
		lp.cv_scoring = par['cv_scoring']
	elif lp.learner_type == 'svm':
		lp.cv_scoring = 'accuracy'

	return lp

if __name__ == "__main__":
	par = read_params(sys.argv)

	f = pd.read_csv(par['inp_f'], sep='\t', header=None, index_col=0, dtype=unicode)
	f = f.T

	if par['out_f']:
		fidout = open(par['out_f'] + '.txt','w')
		fidoutes = open(par['out_f'] + '_estimations.txt','w')
		fidoutroc = open(par['out_f'] + '_roccurve.txt','w')
	else:
		fidout = sys.stdout

	if par['unique']:
		pf = pd.DataFrame([s.split(':') for s in par['unique'].split(',')])

	if par['define']:
		d = pd.DataFrame([s.split(':') for s in par['define'].split(',')])
		l = pd.DataFrame([0]*len(f))
		for i in range(len(d)):
			l[(f[d.iloc[i,1]].isin(d.iloc[i,2:])).tolist()] = d.iloc[i,0]
	else:
		le = prep.LabelEncoder()
		le.fit(f.iloc[:,0])
		l = pd.DataFrame(le.transform(f.iloc[:,0]))

	runs_n = par['runs_n']
	if par ['target']:
		runs_cv_folds = 1
	else:
		runs_cv_folds = par['runs_cv_folds']
	i_tr = pd.DataFrame(True, index=range(len(f.index)), columns=range(runs_n*runs_cv_folds))
	if par['target']:
		i_u = pd.DataFrame(True, index=range(len(f.index)), columns=range(runs_n))
	else:
		if par['unique']:
			i_u = pd.DataFrame(False, index=range(len(f.index)), columns=range(runs_n))
			meta_u = [s for s in f.columns if s in pf.iloc[0,0:].tolist()]
		else:
			i_u = pd.DataFrame(True, index=range(len(f.index)), columns=range(runs_n))
	for j in range(runs_n):
		if par['set_seed']:
			np.random.seed(j)
		if par['target']:
			t = pd.DataFrame([s.split(':') for s in par['target'].split(',')])
			for i in range(len(t)):
				i_tr[j][(f[t.iloc[i,0]].isin(t.iloc[i,1:])).tolist()] = False
		else:
			if par['unique']:
				ii_u = [s-1 for s in (f.loc[np.random.permutation(f.index),:].drop_duplicates(meta_u)).index]
				i_u[j][ii_u] = True
			else:
				ii_u = range(len(f.index))
			if par['set_seed']:
				skf = StratifiedKFold(l.iloc[i_u.values.T[j],0], runs_cv_folds, shuffle=True, random_state=j)
			else:
				skf = StratifiedKFold(l.iloc[i_u.values.T[j],0], runs_cv_folds, shuffle=True)
			for i in range(runs_cv_folds):
				for s in np.where(skf.test_folds == i)[0]:
 					i_tr[j*runs_cv_folds+i][ii_u[s]] = False
	i_tr = i_tr.values.T
	i_u = i_u.values.T

	if par['label_shuffling']:
		np.random.shuffle(l.values)

	feat = [s for s in f.columns if sum([s2 in s for s2 in par['feature_identifier'].split(':')])>0]
	if 'unclassified' in f.columns: feat.append('unclassified')
	f = f.loc[:,feat].astype('float')
	f = (f-f.min())/(f.max()-f.min())

	lp = set_class_params(sys.argv, l)

	fi = []
	clf = []
	p_es = []
	l_es = []
	for j in range(runs_n*runs_cv_folds):
		fi.append(feature_importance(feat, 1.0/len(feat)))
		if lp.feature_selection == 'lasso':
			fi[j] = compute_feature_importance(LassoCV(alphas=lp.fs_grid[0], cv=lp.cv_folds, n_jobs=-1).fit(f.loc[i_tr[j] & i_u[j/runs_cv_folds], fi[j].feat_sel].values, l[i_tr[j] & i_u[j/runs_cv_folds]].values.flatten().astype('int')), feat, fi[j].feat_sel, lp.feature_selection)
		elif lp.feature_selection == 'enet':
			fi[j] = compute_feature_importance(ElasticNetCV(alphas=lp.fs_grid[0], l1_ratio=lp.fs_grid[1], cv=lp.cv_folds, n_jobs=-1).fit(f.loc[i_tr[j] & i_u[j/runs_cv_folds], fi[j].feat_sel].values, l[i_tr[j] & i_u[j/runs_cv_folds]].values.flatten().astype('int')), feat, fi[j].feat_sel, lp.feature_selection)			

		if lp.learner_type == 'rf':
			clf.append(RandomForestClassifier(n_estimators=500, max_depth=None, min_samples_split=2, n_jobs=-1).fit(f.loc[i_tr[j] & i_u[j/runs_cv_folds], fi[j].feat_sel].values, l[i_tr[j] & i_u[j/runs_cv_folds]].values.flatten().astype('int')))			
		elif lp.learner_type == 'svm':
			clf.append(GridSearchCV(SVC(C=1, probability=True), lp.cv_grid, cv=StratifiedKFold(l.iloc[i_tr[j] & i_u[j/runs_cv_folds],0], lp.cv_folds, shuffle=True), scoring=lp.cv_scoring).fit(f.loc[i_tr[j] & i_u[j/runs_cv_folds], fi[j].feat_sel].values, l[i_tr[j] & i_u[j/runs_cv_folds]].values.flatten().astype('int')))
		elif lp.learner_type == 'lasso':
			clf.append(LassoCV(alphas=lp.cv_grid[0], cv=lp.cv_folds, n_jobs=-1).fit(f.loc[i_tr[j] & i_u[j/runs_cv_folds], fi[j].feat_sel].values, l[i_tr[j] & i_u[j/runs_cv_folds]].values.flatten().astype('int')))
		elif lp.learner_type == 'enet':
			clf.append(ElasticNetCV(alphas=lp.cv_grid[0], l1_ratio=lp.cv_grid[1], cv=lp.cv_folds, n_jobs=-1).fit(f.loc[i_tr[j] & i_u[j/runs_cv_folds], fi[j].feat_sel].values, l[i_tr[j] & i_u[j/runs_cv_folds]].values.flatten().astype('int')))
		if (lp.learner_type == 'rf') | (lp.learner_type == 'svm'):
			p_es.append(pd.DataFrame(clf[j].predict_proba(f.loc[-i_tr[j] & i_u[j/runs_cv_folds], fi[j].feat_sel].values)))
			l_es.append(pd.DataFrame([list(p_es[j].iloc[i,:]).index(max(p_es[j].iloc[i,:])) for i in range(len(p_es[j]))]))
		else:
			p_es.append(pd.DataFrame(clf[j].predict(f.loc[-i_tr[j] & i_u[j/runs_cv_folds], fi[j].feat_sel].values)))
			l_es.append(pd.DataFrame([int(p_es[j].iloc[i]>0.5) for i in range(len(p_es[j]))]))
		
	cm = save_results(l, l_es, p_es, i_tr, i_u, len(feat), runs_n, runs_cv_folds)
	fi_f = []
	for j in range(runs_n*runs_cv_folds):
		fi_f.append(compute_feature_importance(clf[j], feat, fi[j].feat_sel, lp.learner_type))

	if lp.learner_type == 'rf':
		for k in lp.cv_grid:
			clf_f = []
			p_es_f = []
			l_es_f = []
			for j in range(runs_n*runs_cv_folds):
				clf_f.append(RandomForestClassifier(n_estimators=500, max_depth=None, min_samples_split=2, n_jobs=-1).fit(f.loc[i_tr[j] & i_u[j/runs_cv_folds], fi_f[j].feat_sel[:k]].values, l[i_tr[j] & i_u[j/runs_cv_folds]].values.flatten().astype('int')))
				p_es_f.append(pd.DataFrame(clf_f[j].predict_proba(f.loc[-i_tr[j] & i_u[j/runs_cv_folds], fi_f[j].feat_sel[:k]].values)))
				l_es_f.append(pd.DataFrame([list(p_es_f[j].iloc[i,:]).index(max(p_es_f[j].iloc[i,:])) for i in range(len(p_es_f[j]))]))
			cm_f = save_results(l, l_es_f, p_es_f, i_tr, i_u, k, runs_n, runs_cv_folds)

	fi_ave = save_average_feature_importance(fi_f, feat)

	if par['out_f']:
		plot_pca(f, l, fi_ave.feat_sel)

		fidout.close()
		fidoutes.close()
		fidoutroc.close()