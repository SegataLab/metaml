#!/usr/bin/env python

import argparse as ap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from mpl_toolkits.mplot3d import Axes3D
from scipy import interp
from sklearn import decomposition
from sklearn import metrics
from sklearn import preprocessing as prep
from sklearn.cross_validation import ShuffleSplit
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
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
		self.cv_folds = []
		self.cv_grid = []
		self.cv_scoring = []

class feature_importance:
	def __init__(self):
		self.feat = []
		self.score = []
		self.std = []

def compute_feature_importance(clf, f, feat):
	fi = feature_importance()
	t = [s.feature_importances_ for s in clf]
	t2 = np.std(t, axis=0)
	t = np.mean(t, axis=0)
	fi.score = sorted(t, reverse=True)
	t = sorted(range(len(t)), key=lambda s: t[s], reverse=True)
	fi.feat = [feat[s] for s in t]
	fi.std = [t2[s] for s in t]
	return fi

def plot_classes(f, l, feat, nf):
	l = l.values.flatten().astype('int')
	f = f[feat[:nf]].values
	
	pca = decomposition.PCA(n_components=3)
	pca = pca.fit(f)
	ft = pca.transform(f)
	
	fig = plt.figure(1, figsize=(4, 3))
	plt.clf()
	plt.cla()
	ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

	for t in range(l.max()+1):
		ax.text3D(ft[l==t, 0].mean(), ft[l==t, 1].mean(), ft[l==t, 2].mean(), 'Class_' + str(t), horizontalalignment='center', bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
	ax.scatter(ft[:, 0], ft[:, 1], ft[:, 2], c=l, cmap=plt.cm.jet)

	ax.w_xaxis.set_ticklabels(['PC1 (' + str(np.round(100*pca.explained_variance_ratio_[0], decimals=2)) + '%)'])
	ax.w_yaxis.set_ticklabels(['PC2 (' + str(np.round(100*pca.explained_variance_ratio_[1], decimals=2)) + '%)'])
	ax.w_zaxis.set_ticklabels(['PC3 (' + str(np.round(100*pca.explained_variance_ratio_[2], decimals=2)) + '%)'])

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

	arg( '-r','--n_runs', default=200, type=int, help="the number of runs\n")
	arg( '-p','--p_test', default=0.1, type=float, help="the dataset proportion to include in the test split\n")

	arg( '-l','--learner_type', choices=['rf','svm'], help="the type of learner (classifier)\n")
	arg( '-f','--cv_folds', type=int, help="the number of cross-validation folds\n")
	arg( '-g','--cv_grid', type=str, help="the parameter grid for cross-validation\n")
	arg( '-s','--cv_scoring', type=str, help="the scoring function for cross-validation\n")
	arg( '-n','--feature_number', type=str, help="the feature number\n")

	arg( '-e','--figure_extension', default='png', type=str, help="the extension of output figure\n")
	
	return vars(parser.parse_args())

def save_fi(fi):
	fidout.write('Feature importance (ranking, name, average, std)\n')
	for k in range(len(fi.feat)):
		fidout.write(str(k) + '\t' + fi.feat[k] + '\t' + str(fi.score[k]) + '\t' + str(fi.std[k]) + '\n')

def save_results(l, l_es, p_es, i_tr, i_u, f):
	n_runs = len(l_es)
	cm = class_metrics()

	for j in range(n_runs):
		l_ = pd.DataFrame([l.loc[i] for i in l[-i_tr[j] & i_u[j]].index]).values.flatten().astype('int')
		l_es_ = l_es[j].values.flatten().astype('int')
		p_es_pos_ = p_es[j].loc[:,1].values

		cm.accuracy.append(metrics.accuracy_score(l_, l_es_))
		cm.f1.append(metrics.f1_score(l_, l_es_, pos_label=None, average='weighted'))
		cm.precision.append(metrics.precision_score(l_, l_es_, pos_label=None, average='weighted'))
		cm.recall.append(metrics.recall_score(l_, l_es_, pos_label=None, average='weighted'))
		if max(l_)==1:
			cm.auc.append(metrics.roc_auc_score(l_, p_es_pos_))
			cm.roc_curve.append(metrics.roc_curve(l_, p_es_pos_))
		cm.confusion_matrix.append(metrics.confusion_matrix(l_, l_es_))

	fidout.write('#samples\t' + str(sum(sum(i_u))/len(i_u)))
	fidout.write('\n#features\t' + str(len(f.iloc[0,:])))
	fidout.write('\n#runs\t' + str(len(l_es)))

	fidout.write('\naccuracy\t' + str(np.mean(cm.accuracy)) + '\t' + str(np.std(cm.accuracy)))
	fidout.write('\nf1\t' + str(np.mean(cm.f1)) + '\t' + str(np.std(cm.f1)) + ')')
	fidout.write('\nprecision\t' + str(np.mean(cm.precision)) + '\t' + str(np.std(cm.precision)))
	fidout.write('\nrecall\t' + str(np.mean(cm.recall)) + '\t' + str(np.std(cm.recall)))
	fidout.write('\nauc\t' + str(np.mean(cm.auc)) + '\t' + str(np.std(cm.auc)))
	fidout.write('\nconfusion matrix')
	for i in range(len(cm.confusion_matrix[0])):
		for i2 in range(len(cm.confusion_matrix[0][i])):
			fidout.write('\t' + str(np.sum([cm.confusion_matrix[j][i][i2] for j in range(n_runs)])))
		fidout.write('\n')
	fidoutroc.write('#features\t' + str(len(f.iloc[0,:])) + '\n')		
	for j in range(n_runs):
		for i in range(len(cm.roc_curve[j])):
			for i2 in range(len(cm.roc_curve[j][i])):
				fidoutroc.write(str(cm.roc_curve[j][i][i2]) + '\t')
			fidoutroc.write('\n')

	return cm

def set_class_params(args):
	lp = class_params()

	if par['learner_type']:
		lp.learner_type = par['learner_type']
	else:
		lp.learner_type = 'rf'

	if par['cv_folds']:
		lp.cv_folds = int(par['cv_folds'])
	elif lp.learner_type == 'svm':
		lp.cv_folds = 5

	if par['cv_grid']:
		lp.learner_type = par['cv_grid']
	elif lp.learner_type == 'svm':
		if par['feature_identifier'] == 'k__':
			# lp.cv_grid = [ {'C': [2**s for s in range(-5,16,2)], 'kernel': ['linear']}, {'C': [2**s for s in range(-5,16,2)], 'gamma': [2**s for s in range(3,-15,-2)], 'kernel': ['rbf']} ]
			lp.cv_grid = [ {'C': [1, 10, 100, 1000], 'kernel': ['linear']}, {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']} ]
		else:
			lp.cv_grid = [ {'C': [1, 10, 100, 1000], 'kernel': ['linear']}]
	
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

	if par['label_shuffling']:
		np.random.shuffle(l.values)

	if par ['target']:
		n_runs = 1
		i_u = pd.DataFrame(True, index=range(len(f.index)), columns=range(n_runs))
		i_tr = pd.DataFrame(True, index=range(len(f.index)), columns=range(n_runs))
		t = pd.DataFrame([s.split(':') for s in par['target'].split(',')])		
		for i in range(len(t)):
			i_tr[(f[t.iloc[i,0]].isin(t.iloc[i,1:])).tolist()] = False
	else:
		n_runs = par['n_runs']
		i_tr = pd.DataFrame(True, index=range(len(f.index)), columns=range(n_runs))
		if par['unique']:
			i_u = pd.DataFrame(False, index=range(len(f.index)), columns=range(n_runs))
			meta_u = [s for s in f.columns if s in pf.iloc[0,0:].tolist()]
			for j in range(n_runs):
				ii_u = [s-1 for s in (f.loc[np.random.permutation(f.index),:].drop_duplicates(meta_u)).index]
				i_u[j][ii_u] = True
				rs = [s[1] for s in ShuffleSplit(len(ii_u), n_iter=1, test_size=par['p_test'])]
				for s in np.nditer(rs):
					i_tr[j][ii_u[s]] = False
		else:
			i_u = pd.DataFrame(True, index=range(len(f.index)), columns=range(n_runs))
			rs = [s[1] for s in ShuffleSplit(len(l), n_iter=n_runs, test_size=par['p_test'])]
			for j in range(n_runs):
				i_tr[j][rs[j]] = False
	i_tr = i_tr.values.T
	i_u = i_u.values.T

	feat = [s for s in f.columns if sum([s2 in s for s2 in par['feature_identifier'].split(':')])>0]
	if 'unclassified' in f.columns: feat.append('unclassified')
	f = f.loc[:,feat].astype('float')
	f = (f-f.min())/(f.max()-f.min())

	if par['feature_number']:
		nf = [int(s) for s in par['feature_number'].split(':')]
	else:
		nf = [f.shape[1] for s in range(1)]

	lp = set_class_params(sys.argv)

	clf = []
	p_es = []
	l_es = []
	for j in range(n_runs):
		if lp.learner_type=='svm':
			clf.append(GridSearchCV(SVC(C=1, probability=True), lp.cv_grid, cv=StratifiedKFold(l.iloc[i_tr[j] & i_u[j],0], lp.cv_folds), scoring=lp.cv_scoring).fit(f[i_tr[j] & i_u[j]].values, l[i_tr[j] & i_u[j]].values.flatten().astype('int')))
			p_es.append(pd.DataFrame(clf[-1].predict_proba(f[-i_tr[j] & i_u[j]].values)))
		if lp.learner_type=='rf':
			clf.append(RandomForestClassifier(n_estimators=500, max_depth=None, min_samples_split=1, random_state=0).fit(f[i_tr[j] & i_u[j]].values, l[i_tr[j] & i_u[j]].values.flatten().astype('int')))
			p_es.append(pd.DataFrame(clf[-1].predict_proba(f[-i_tr[j] & i_u[j]].values)))
		l_es.append(pd.DataFrame([list(p_es[j].iloc[i,:]).index(max(p_es[j].iloc[i,:])) for i in range(len(p_es[j]))]))
	cm = save_results(l, l_es, p_es, i_tr, i_u, f)

	if lp.learner_type=='rf':
		fi_ave = compute_feature_importance(clf, f, feat)
		fidout.write('\n')
		if par['feature_number']:
			fi = []
			for k in nf:
				clf_f = []
				p_es_f = []
				l_es_f = []
				for j in range(n_runs):
					if k==nf[0]:
						fi.append(compute_feature_importance(clf[j], f, feat))
					clf_f.append(RandomForestClassifier(n_estimators=500, max_depth=None, min_samples_split=1, random_state=0).fit(f.loc[i_tr[j] & i_u[j], fi[j].feat[:k]].values, l[i_tr[j] & i_u[j]].values.flatten().astype('int')))
					p_es_f.append(pd.DataFrame(clf_f[j].predict_proba(f.loc[-i_tr[j] & i_u[j],fi[j].feat[:k]].values)))
					l_es_f.append(pd.DataFrame([list(p_es_f[j].iloc[i,:]).index(max(p_es_f[j].iloc[i,:])) for i in range(len(p_es_f[j]))]))
				cm_f = save_results(l, l_es_f, p_es_f, i_tr, i_u, f.loc[:,fi_ave.feat[:k]])
				fidout.write('\n')
		if par['out_f']:
			plot_classes(f, l, fi_ave.feat, nf[-1])
		save_fi(fi_ave)

	if par['out_f']:
		fidout.close()
		fidoutroc.close()