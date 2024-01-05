#!/usr/bin/env python


import time
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
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
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

		# self.sensitivity = []
		# self.specificity = []
		# mcc  


class class_params:
	def __init__(self):
		self.learner_type = []
		self.feature_selection = []
		self.cv_folds = []
		self.cv_grid = []
		self.cv_scoring = []
		self.fs_grid = []

		#self.refine = []
		#self.refine_grid = []


class feature_importance:
	def __init__(self, feat, p):
		self.feat_sel = feat
		self.imp = np.array([p]*len(feat))


def compute_feature_importance(el, feat, feat_sel, ltype):
	fi = feature_importance(feat, 0.0)
	if ltype in ['rf', 'gb']:
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
		ax.scatter(ft[l==i, 0], ft[l==i, 1], c=l[l==i], s=200, \
		cmap=plt.cm.jet, vmin=min(l), vmax=max(l), edgecolors='None', alpha=0.6)
		ax.text(ft[l==i, 0].mean(), ft[l==i, 1].mean(), 'Class ' + str(i), \
		horizontalalignment='center', bbox=dict(alpha=0.5, edgecolor='w', facecolor='w'))

	ax.set_xlabel('PC1 (' + str(np.round(100*pca.explained_variance_ratio_[0], decimals=2)) + '%)')
	ax.set_ylabel('PC2 (' + str(np.round(100*pca.explained_variance_ratio_[1], decimals=2)) + '%)')
	ax.set_xlim([min(ft[:, 0]),max(ft[:, 0])])
	ax.set_ylim([min(ft[:, 1]),max(ft[:, 1])])	
	
	fig.savefig(par['out_f'] + '_pca.' + par['figure_extension'], bbox_inches='tight')


def read_params():
	parser = ap.ArgumentParser(\
	    description='MetAML - Metagenomic prediction Analysis based on Machine Learning')
	arg = parser.add_argument
	arg( 'inp_f', metavar='INPUT_FILE', nargs='?', default=sys.stdin, type=str\
		, help="the input dataset file [stdin if not present]")
	arg( 'out_f', metavar='OUTPUT_FILE', nargs='?', default=None, type=str\
		, help="the output file [stdout if not present]")

	arg( '-z','--feature_identifier', default='k__', type=str, help="the feature identifier\n")
	arg( '-d','--define', type=str, help="define the classification problem\n")
	arg( '-t','--target', type=str, help="define the target domain\n")

	arg( '-x','--reinforcement', type=str, help="define the domain to support the cross-validation")

	arg( '-u','--unique', type=str, help="the unique samples to select\n")
	arg( '-b','--label_shuffling', action='store_true', help="label shuffling\n")

	arg( '-r','--runs_n', default=20, type=int, help="the number of runs\n")
	arg( '-p','--runs_cv_folds', default=10, type=int, help="the number of cross-validation folds per run\n")
	arg( '-w','--set_seed', action='store_true', help="setting seed\n")
	arg( '-l','--learner_type', choices=['rf','lsvm','svm','lasso','enet','gb'], default='rf', help='the type of learner/classifier\n')
	arg( '-i','--feature_selection', choices=['lasso','enet'], help="the type of feature selection\n")

	arg( '-f','--cv_folds', type=int, help="the number of cross-validation folds for model selection\n")
	arg( '-g','--cv_grid', type=str, help="the parameter grid for model selection\n")
	arg( '-s','--cv_scoring', default='roc_auc', type=str, help="the scoring function for model selection\n")
	arg( '-j','--fs_grid', type=str, help="the parameter grid for feature selection\n")

	## random forest options
	arg( '-c','--rf_criterion', type=str, choices=['gini', 'entropy'], default='entropy', \
	    help='Impurity criterion (random forest)')
	arg( '-mf','--rf_max_features'\
	    , choices=['0.001', '0.01', '0.1', '0.2', '0.3', '0.5', '0.4', '0.6', '1.0'\
	    , '100', 'auto', 'sqrt', '0.33', None, 'log2','10'], default=0.3, \
	    help='Feature sample/percentage (random forest)')

	arg( '-nt','--number_of_trees', type=int, default=1000, help='# of estimator trees (random forest)')
	arg( '-nsl','--number_sample_per_leaf', type=int, default=1, help='minimum # sample per leaf (random forest)')
	arg( '-oob','--oob_score', action='store_true', help='Enable out-of-bag choice in random forest')
	arg( '-df','--disable_features', action='store_true', help='Doesn\'t perform the features selection' +\
             'which follows std random forest (random forest)')
	arg( '-cc', '--choose_cut', type=str, default=None, \
	    help='comma-separated list of numbers which will substitute the std features cuts (10,20,...,150)')
	arg( '-wc', '--weight_classes', default=None, type=float, nargs=2)

	arg( '-hv','--how_verbose', default=1, type=int, choices=[0,1,2])
	arg( '-e','--figure_extension', default='png', type=str, help="the extension of output figure\n")
	arg( '-nc', '--ncores', type=int, default=10, help='-1 set to all the available cores.')
	arg( '-lk','--linear_kernel', action='store_true')
	arg( '--no_norm', action='store_true', help='Disable per-sample normalisation')

	arg( '-ovec', '--objective_vector', type=str, default=None, \
	    help='Like a DEFINED, but it only replaces the final vector on which to predict on')

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
	[fidout.write(str(s) + '\t' + feat[t[s]] + '\t' + str(fi_ave.imp[t[s]]) \
		+ '\t' + str(fi_ave_std[t[s]]) + '\n') for s in range(len(t))]

	return fi_ave




def save_results(l, l_es, p_es, i_tr, i_u, nf, runs_n, runs_cv_folds):

	n_clts = len(np.unique(l.values.flatten().astype('int')))
	cm = class_metrics()

	if par['out_f']:
		fidoutes.write('#features\t' + str(nf) + '\n')
		if n_clts == 2:
			fidoutroc.write('#features\t' + str(nf) + '\n')	

	for j in range( runs_n * runs_cv_folds ):
		l_ = pd.DataFrame([l.loc[i] for i in l[~i_tr[j] & i_u[j//runs_cv_folds]].index]).values.flatten().astype('int')

		l_es_ = l_es[j].values.flatten().astype('int')
		#print(p_es[j].sum())
		#print(j, p_es[j].shape, "J e p_es J")

		if (lp.learner_type == 'rf') | (lp.learner_type.endswith('svm')) | (lp.learner_type=='gb'):
			p_es_pos_ = p_es[j].loc[:,1].values
		else:
			p_es_pos_ = p_es[j].loc[:,0].values
		ii_ts_ = [i for i in range(len(i_tr[j])) if i_tr[j][i]==False]

		cm.accuracy.append(metrics.accuracy_score(l_, l_es_))
		cm.f1.append(metrics.f1_score(l_, l_es_, pos_label=None, average='weighted'))
		cm.precision.append(metrics.precision_score(l_, l_es_, pos_label=None, average='weighted'))
		cm.recall.append(metrics.recall_score(l_, l_es_, pos_label=None, average='weighted'))

		if len(np.unique(l_)) in [ n_clts, n_clts-1 ]:
			if n_clts == 2:

				try: cm.auc.append(metrics.roc_auc_score(l_, p_es_pos_))
				except ValueError: cm.auc.append(0.0)

				cm.roc_curve.append(metrics.roc_curve(l_, p_es_pos_))
				fidoutroc.write('run/fold\t' + str(j//runs_cv_folds) + '/' + str(j%runs_cv_folds) + '\n')
				for i in range(len(cm.roc_curve[-1])):
					for i2 in range(len(cm.roc_curve[-1][i])):
						fidoutroc.write(str(cm.roc_curve[-1][i][i2]) + '\t')
					fidoutroc.write('\n')

			cf = metrics.confusion_matrix(l_, l_es_, labels=np.unique(l.astype('int')))
			cm.confusion_matrix.append(cf)
			##	metrics.confusion_matrix(l_, l_es_, labels=np.unique(l.astype('int'))))
			#cm.sensitivity.append( cf[1,1] / float( cf[1,1] + cf[1,0] ) )
			#cm.specificity.append( cf[0,0] / float( cf[0,0] + cf[0,1] ) )

		if par['out_f']:
			fidoutes.write('run/fold\t' + str(j//runs_cv_folds) + '/' + str(j%runs_cv_folds))
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

        #fidout.write('\nsensitivity\t' + str(np.mean(cm.sensitivity)) + '\t' + str(np.std(cm.sensitivity)))
        #fidout.write('\nspecificity\t' + str(np.mean(cm.specificity)) + '\t' + str(np.std(cm.specificity)))

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

	lp.refine_grid = [{'C': [1,1000], 'kernel':['linear']}, {'C': [1, 1000], 'gamma': [10, 1, 0.1, 0.01, 0.001, 0.0001], 'kernel':['rbf']}]

	if par ['feature_selection']:
		lp.feature_selection = par['feature_selection']
	else:
		lp.feature_selection = 'none'

	if par['cv_folds']:
		lp.cv_folds = int(par['cv_folds'])
	else:
		lp.cv_folds = 10

	if par['cv_grid']:
		lp.cv_grid = eval(par['cv_grid'])

	elif lp.learner_type == 'rf':
		lp.cv_grid = [{'max_features': [0.33, 'auto']}]
	#	lp.cv_grid = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 32]
		# 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 125, 150, 175, 200]

	elif lp.learner_type == 'svm':
		#if par['feature_identifier'] == 'k__':
		lp.cv_grid = [ {'C': [2**s for s in range(-5,16,2)], 'kernel': ['linear']}, \
		    {'C': [2**s for s in range(-5,16,2)], 'gamma': [2**s for s in range(3,-15,-2)], 'kernel': ['rbf']} ]
		#	lp.cv_grid = [ {'C': [1, 10, 100, 1000], 'kernel': ['linear']}, \
		#   {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']} ]

	elif lp.learner_type == 'lsvm':
		lp.cv_grid = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']}]

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
	
	lp.cv_scoring = par['cv_scoring']

	return lp



if __name__ == "__main__":

	par = read_params()
 
	if par['rf_max_features'] in ['0.001', '0.01', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.33','100','10','1.0']:
		par['rf_max_features'] = float(par['rf_max_features'])

	f = pd.read_csv(par['inp_f'], sep='\t', header=None, index_col=0) #, dtype=unicode)
	f = f.T

	if par['reinforcement']:
		t = pd.DataFrame([s.split(':') for s in par['reinforcement'].split(',')]).fillna(np.nan)
		f_re = f.copy()

		for i in range(len(t)): 
			f = f[~f[ t.iloc[i,0] ].isin( t.iloc[i,1:]) ]

		f_re = f_re[ ~f_re.index.isin(f.index) ]

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

		if par['objective_vector']: 
			d_ob = pd.DataFrame([s.split(':') for s in par['objective_vector'].split(',')])
			l_ob = pd.DataFrame([0]*len(f))

		for i in range(len(d)):
			l[(f[d.iloc[i,1]].isin(d.iloc[i,2:])).tolist()] = d.iloc[i,0]
			if par['objective_vector']:
				l_ob[(f[d_ob.iloc[i,1]].isin(d_ob.iloc[i,2:])).tolist()] = d_ob.iloc[i,0]

		if par['reinforcement']:
			l_re = pd.DataFrame([0]*len(f_re))
			for i in range(len(d)):
			      ##l   [(f   [d.iloc[i,1]].isin(d.iloc[i,2:])).tolist()] = d.iloc[i,0] 
				l_re[(f_re[d.iloc[i,1]].isin(d.iloc[i,2:])).tolist()] = d.iloc[i,0]

	else:
		le = prep.LabelEncoder()
		le.fit(f.iloc[:,0])
		l = pd.DataFrame(le.transform(f.iloc[:,0])).astype('int')
		l_ob = pd.DataFrame(le.transform(f.iloc[:,0])).astype('int')

	runs_n = par['runs_n']

	if par ['target']:
		runs_cv_folds = 1
	else:
		runs_cv_folds = par['runs_cv_folds']

	i_tr = pd.DataFrame(True, index=range(len(f.index)), columns=range(runs_n*runs_cv_folds))

	if par['objective_vector']:
		i_tr_ob = pd.DataFrame(True, index=range(len(f.index)), columns=range(runs_n*runs_cv_folds))

	if par['target']:
		i_u = pd.DataFrame(True, index=range(len(f.index)), columns=range(runs_n))
	else:
		if par['unique']:
			i_u = pd.DataFrame(False, index=range(len(f.index)), columns=range(runs_n))
			meta_u = [s for s in f.columns if s in pf.iloc[0,0:].tolist()]
		else:
			i_u = pd.DataFrame(True, index=range(len(f.index)), columns=range(runs_n))

	if par['reinforcement']:
		i_tr_re = pd.DataFrame(True, index=range(len(f_re.index)), columns=range(runs_n*runs_cv_folds))

		if par['target']:
			i_u_re = pd.DataFrame(True, index=range(len(f_re.index)), columns=range(runs_n))

		if par['unique']:
			i_u_re = pd.DataFrame(False, index=range(len(f_re.index)), columns=range(runs_n))
			meta_u = [s for s in f_re.columns if s in pf.iloc[0,0:].tolist()]
		else:
			i_u_re = pd.DataFrame(True, index=range(len(f_re.index)), columns=range(runs_n))

	for j in range(runs_n):
		if par['set_seed']:
			np.random.seed(j)

		if par['target']:
			t = pd.DataFrame([s.split(':') for s in par['target'].split(',')])
			for i in range(len(t)):
				i_tr[j][(f[t.iloc[i,0]].isin(t.iloc[i,1:])).tolist()] = False

				if par['objective_vector']:
					i_tr_ob[j][(f[t.iloc[i,0]].isin(t.iloc[i,1:])).tolist()] = False

		else:

			if par['unique']:
				ii_u = [s-1 for s in (f.loc[np.random.permutation(f.index),:].drop_duplicates(meta_u)).index]
				i_u[j][ii_u] = True
			else:
				ii_u = range(len(f.index))

			skf = StratifiedKFold(n_splits = runs_cv_folds, shuffle = True, random_state = (j if par['set_seed'] else None))	
			lll = np.array(l.iloc[i_u.values.T[j], 0].values, dtype=np.float64)
 
			skf_split = skf.split(np.array([[0, 0] for q in range(len(lll))], dtype=np.float64), lll)
			test_folds = [tf[1] for tf in skf_split] # [train_index,test_index in skf_split]]

			for i in range(runs_cv_folds):
				for s in test_folds[i]:
					i_tr[ j*runs_cv_folds + i ][ ii_u[ s ] ] = False ## False == TEST

	if par['reinforcement']:

		i_tr = pd.concat([ i_tr, i_tr_re ]).reset_index(drop=True)
		i_u = pd.concat([ i_u, i_u_re ]).reset_index(drop=True)

		f = pd.concat([f, f_re]).reset_index(drop=True).fillna(0.0)
		l = pd.concat([ l, l_re ]).reset_index(drop=True)
		
		#else:	## agumentation
		#	t = pd.DataFrame([s.split(':') for s in par['augmentation'].split(',')])
		#	
		#
		#	#for i in range(len(t)):
                #        #        i_tr_ag[j][(f[t.iloc[i,0]].isin(t.iloc[i,1:])).tolist()] = False
		#	#	i_tr_cr[j][(f[t.iloc[i,0]].isin(t.iloc[i,1:])).tolist()] = False
		#
		#
		#	skf = StratifiedKFold(n_splits = runs_cv_folds, shuffle = True, random_state = (j if par[ 'set_seed' ] else None))
		#
		#	a_v = np.array(l.iloc[i_u.values.T[j], 0].values, dtype=np.float64)
		#
		#		#for i in range(len(t)):
		#		#	a_v = a_v[ f[t.iloc[i,0]].isin(t.iloc[i,1:]).values.nonzero()[0] ]
		#
		#		# print(a_v, a_v.shape, " MINCHIE ")
		#
		#	skf_split = skf.split(np.array([[0, 0] for q in range(len(a_v))], dtype=np.float64), a_v)
		#	  
		#		#for tf in skf_split:
		#		#	i = 0
		#		#	print( f[t.iloc[i,0]].isin(t.iloc[i,1:]).values.shape, tf[1][ f[t.iloc[i,0]].isin(t.iloc[i,1:]).values.nonzero()[0] ].shape , "MINCHIA" )
 		#
		#	t_fs = [ tf[1] for tf in skf_split ]
		#	tt_fs = []
  		#
		#	for q,tf in enumerate(t_fs):
		#		for i in range(len(t)):
		#			tt_fs.append( tf[ f[t.iloc[i,0]].isin(t.iloc[i,1:]).values.nonzero()[0] ] )
		#			for x in tt_fs:
		#				print(len([xx for xx in x if xx<343]))
 		#
		#				#print( len(tf), len(tf[ f[t.iloc[i,0]].isin(t.iloc[i,1:]).values.nonzero()[0] ]), len(t_fs[q]), " e almeno pou piccolo " )
		#
		#	exit(1)
		#
                #                           ### i_tr[j][(f[t.iloc[i,0]].isin(t.iloc[i,1:])).tolist()] = False
		#			 
		#	t_fs = tt_fs
		#
		#	for i in range(runs_cv_folds):
		#		for s in t_fs[i]:
		#			i_tr[j*runs_cv_folds + i ][ii_u[ s ]] = False
		#
		#
		#	for e,k in enumerate(i_tr):
		#		print(e, np.invert(i_tr[k]).sum(), i_tr[k].sum(), np.invert(i_tr[k]).sum() + i_tr[k].sum(), " TEST/TRAIN/TOT ")
			
	i_tr = i_tr.values.T
	i_u = i_u.values.T

	if par['label_shuffling']:
		np.random.shuffle(l.values)

	#for s in f.columns:
		#print(par['feature_identifier'].split(':'), s)
                #print(s, [s2 in s for s2 in par['feature_identifier'].split(':')])
		#if sum([s2 in s for s2 in par['feature_identifier'].split(':')])>0:
		#	try:
		#	 	print(sum(f[s].astype(float)))
		#	except:
		#		print(f[s].unique().tolist())

	feat = [s for s in f.columns if ((sum([s2 in s for s2 in par['feature_identifier'].split(':')])>0) and (sum(f[s].astype(float))>0.))]
	if 'unclassified' in f.columns: feat.append('unclassified')
	f = f.loc[:, feat].astype('float')
		 
	#if par['augmentation']:
	#	feat_ag = [s for s in f_ag.columns if ((sum([s2 in s for s2 in par['feature_identifier'].split(':')])>0) and (s in feat))]
	#	f_ag = f_ag.loc[:, feat_ag].astype('float')
	#	f_ag = f_ag.div(f_ag.sum(axis=1)) * 100.

	#if not par['no_norm']:
		#if not par['augmentation']:
	f = (f-f.min())/(f.max()-f.min())
		#else:
		#	print(f.min(), "f min")
		#	print(f_ag.min(), "f ag min")
		#	print(f.max(), "f max")
		#	print(f_ag.max(), "f ag max")

		#	f = (  f  -  np.min([ f.min(), f_ag.min() ])  )/(  np.max([f.max(), f_ag.max()])  -  np.min([f.min(), f_ag.min()])  )
		#	f_ag = (  f_ag  -  np.min([f.min(), f_ag.min()]) ) / (  np.max([f.max(), f_ag.max()])  -  np.min([f.min(), f_ag.min()])  )

	lp = set_class_params(sys.argv, l)

	fi = []
	clf = []
	p_es = []
	l_es = []

	global_time = time.time()

	for j in range(runs_n*runs_cv_folds):
		start_run_time = time.time()
		fi.append(feature_importance(feat, 1.0/len(feat)))

		tr_st = f.loc[i_tr[j] & i_u[j//runs_cv_folds], fi[j].feat_sel].values
		tr_st_t = l[i_tr[j] & i_u[j//runs_cv_folds]].values.flatten().astype('int')

		if lp.feature_selection == 'lasso':
			fi[j] = compute_feature_importance(LassoCV(\
			alphas=lp.fs_grid[0], cv=lp.cv_folds, n_jobs=par['ncores'], verbose=par['how_verbose']).fit(\
			tr_st, tr_st_t), feat, fi[j].feat_sel, lp.feature_selection)

		elif lp.feature_selection == 'enet':
			fi[j] = compute_feature_importance(ElasticNetCV(\
			alphas=lp.fs_grid[0], l1_ratio=lp.fs_grid[1], cv=lp.cv_folds, n_jobs=par['ncores']).fit(\
			tr_st, tr_st_t), feat, fi[j].feat_sel, lp.feature_selection)

		if lp.learner_type == 'rf':

			if not par['rf_max_features']:
				hypers = GridSearchCV(\
					RandomForestClassifier(\
					  n_estimators=par['number_of_trees']\
					, min_samples_leaf=par['number_sample_per_leaf']\
					, criterion='entropy'\
					, max_depth=None\
					, min_samples_split=2\
					, n_jobs=10\
					, verbose=par['how_verbose'], oob_score=par['oob_score'])\
					, lp.cv_grid, cv=StratifiedKFold(\
					  l.iloc[i_tr[j] & i_u[j/runs_cv_folds],0], lp.cv_folds, shuffle=True)\
					, scoring=lp.cv_scoring, refit=False).fit( tr_st, tr_st_t )
	
				clf.append(RandomForestClassifier(\
					n_estimators=par['number_of_trees'], criterion=par['rf_criterion']\
					, max_features=hypers.best_params_['max_features']\
					, oob_score=par['oob_score'], max_depth=None, min_samples_split=2\
					, n_jobs=par['ncores'], verbose=par['how_verbose']\
					, min_samples_leaf=par['number_sample_per_leaf']).fit( tr_st, tr_st_t ))

			else:
				clf.append(RandomForestClassifier(\
					n_estimators=par['number_of_trees']\
					, criterion=par['rf_criterion']\
					, max_features=par['rf_max_features']\
					, oob_score=par['oob_score'], max_depth=None\
					, min_samples_split=2, n_jobs=par['ncores']\
					, verbose=par['how_verbose']\
					, min_samples_leaf=par['number_sample_per_leaf']\
					, class_weight='balanced').fit( tr_st, tr_st_t ))

		elif lp.learner_type == "gb":
                    clf.append(GradientBoostingClassifier(\
                                        n_estimators=par['number_of_trees']\
                                        , loss="deviance", learning_rate=0.1 \
                                        , max_features=par['rf_max_features']\
                                        , criterion='friedman_mse', max_depth=3\
                                        , min_samples_split=2 \
                                        , verbose=par['how_verbose']\
                                        , min_samples_leaf=par['number_sample_per_leaf']\
                                        ).fit(  tr_st, tr_st_t ))

		elif lp.learner_type.endswith('svm'):
			clf.append(GridSearchCV(\
                                SVC(probability=True, verbose=bool(par['how_verbose']))\
                                , lp.cv_grid, cv=StratifiedKFold(l.iloc[i_tr[j] & i_u[j/runs_cv_folds],0]\
				, lp.cv_folds, shuffle=True)\
                                , scoring=lp.cv_scoring).fit( tr_st, tr_st_t ))

		elif lp.learner_type == 'lasso':
			clf.append(LassoCV(alphas=lp.cv_grid[0], cv=lp.cv_folds, n_jobs=par['ncores'], verbose=par['how_verbose']).fit(\
				tr_st, tr_st_t ))

		elif lp.learner_type == 'enet':
			clf.append(ElasticNetCV(\
			alphas=lp.cv_grid[0], l1_ratio=lp.cv_grid[1], cv=lp.cv_folds, n_jobs=-1).fit( tr_st, tr_st_t ))
 
		if (lp.learner_type == 'rf') | (lp.learner_type.endswith('svm')) | (lp.learner_type=='gb'):
			p_es.append(pd.DataFrame(clf[j].predict_proba(f.loc[~i_tr[j] & i_u[j//runs_cv_folds], fi[j].feat_sel].values)))
			l_es.append(pd.DataFrame([list(p_es[j].iloc[i,:]).index(max(p_es[j].iloc[i,:])) for i in range(len(p_es[j]))]))

		elif lp.learner_type == 'precomp':
			p_es.append(pd.DataFrame(f.loc[~i_tr[j] & i_u[j//runs_cv_folds], fi[j].feat_sel].values))
			l_es.append(pd.DataFrame([int(p_es[j].iloc[i]>0.5) for i in range(len(p_es[j]))]))

		else:
			p_es.append(pd.DataFrame(clf[j].predict(f.loc[~i_tr[j] & i_u[j//runs_cv_folds], fi[j].feat_sel].values)))
			l_es.append(pd.DataFrame([int(p_es[j].iloc[i]>0.5) for i in range(len(p_es[j]))]))

		elapsed_run_time = time.time() - start_run_time
		if par['how_verbose'] > 0:
			print('%i run-time: %.4f sec.' %(j, float(elapsed_run_time)))

	#print(p_es[-1], "questa e p_es")	
	#print(l_es[-1], "questa e l_es")
	#print(l, " questa e l")
	
	cm = save_results(l if not par['objective_vector'] else l_ob, \
		l_es, p_es, i_tr if not par['objective_vector'] else i_tr_ob, \
		i_u, len(feat), runs_n, runs_cv_folds)

	global_elapsed = time.time() - global_time
	if par['how_verbose'] > 0:
		print ('global-time: %.4f sec.' %(global_elapsed))

	## from here on, if you used ranfom forest and you didn't specify
	## disable_features (which is useful in case of huge of very large database)
	## you have extracted a ranking of the most predictive features which
	## is averaged over '# folds * # runs' cicles. The testing sets are at each 
        ## cycle excluded, so you can use this set of selected features without worrying
	if lp.learner_type in ['rf', 'gb']:
		if not par['disable_features']:
			fi_f = []
			for j in range(runs_n*runs_cv_folds):
                               	fi_f.append(compute_feature_importance(clf[j], feat, fi[j].feat_sel, lp.learner_type))

			if not par['choose_cut']:
				if par['feature_identifier'] != 'UniRef90': 
					steps = [1,2,4,8,16,32,64] 
				elif par['feature_identifier'] == 'UniRef90': 
					steps = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096] 

			else:
				steps = list(map(int, par['choose_cut'].split(',')))
 
			if lp.learner_type == 'rf':
				for k in steps:
					clf_f = []
					p_es_f = []
					l_es_f = []

					for j in range(runs_n*runs_cv_folds):
						clf_f.append(\
							RandomForestClassifier(n_estimators=par['number_of_trees']
							, criterion=par['rf_criterion']\
							, max_features=(k if par['feature_identifier'] != 'UniRef90' else (k if k<=128 else 0.3))\
							, max_depth=None\
							, min_samples_split=2, oob_score=par['oob_score']\
							, min_samples_leaf=par['number_sample_per_leaf']\
							, n_jobs=par['ncores'], verbose=par['how_verbose']\
							, class_weight='balanced')\
							.fit(f.loc[i_tr[j] & i_u[j//runs_cv_folds], fi_f[j].feat_sel[:k] ].values\
						, l[i_tr[j] & i_u[j//runs_cv_folds]].values.flatten().astype('int')))

						p_es_f.append(pd.DataFrame(clf_f[j].predict_proba(f.loc[~i_tr[j] & i_u[j//runs_cv_folds]\
							, fi_f[j].feat_sel[:k]].values)))
						l_es_f.append(pd.DataFrame([list(p_es_f[j].iloc[i,:]).index(max(p_es_f[j].iloc[i,:])) \
						for i in range(len(p_es_f[j]))]))

					cm_f = save_results(l if not par['objective_vector'] else l_ob, \
						l_es_f, p_es_f, i_tr if not par['objective_vector'] else i_tr_ob, \
						i_u, k, runs_n, runs_cv_folds)

			elif lp.learner_type == 'gb':
				for k in steps:
					clf_f = []
					p_es_f = []
					l_es_f = []

					for j in range(runs_n*runs_cv_folds):
						clf_f.append(\
							GradientBoostingClassifier(\
                                        		n_estimators=par['number_of_trees']\
                                        		, loss="deviance", learning_rate=0.1 \
                                        		, max_features=par['rf_max_features']\
                                        		, criterion='friedman_mse', max_depth=3\
                                        		, min_samples_split=2 \
                                        		, verbose=par['how_verbose']\
                                        		, min_samples_leaf=par['number_sample_per_leaf']\
                                        	).fit(\
                                			f.loc[i_tr[j] & i_u[j//runs_cv_folds], fi_f[j].feat_sel[:k] ].values\
                                			, l[i_tr[j] & i_u[j//runs_cv_folds]].values.flatten().astype(int)))

						p_es_f.append(pd.DataFrame(clf_f[j].predict_proba(f.loc[~i_tr[j] & i_u[j//runs_cv_folds]\
                                                                , fi_f[j].feat_sel[:k]].values)))
						l_es_f.append(pd.DataFrame([list(p_es_f[j].iloc[i,:]).index(max(p_es_f[j].iloc[i,:])) \
                                                        for i in range(len(p_es_f[j]))]))

					cm_f = save_results(l if not par['objective_vector'] else l_ob, \
                                                        l_es_f, p_es_f, i_tr if not par['objective_vector'] else i_tr_ob, \
                                                        i_u, k, runs_n, runs_cv_folds)
                        
			fi_ave = save_average_feature_importance(fi_f, feat)
 
	if par['out_f']:
		if lp.learner_type in ['rf', 'gb']:
			if not par['disable_features']:
				plot_pca(f, l if not par['objective_vector'] else l_ob, fi_ave.feat_sel)
 
		fidout.close()
		fidoutes.close()
		fidoutroc.close()
