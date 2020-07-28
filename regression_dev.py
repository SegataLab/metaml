#!/shares/CIBIO-Storage/CM/scratch/users/paolo.manghi/anaconda3/bin/python

import time
import sys
import argparse as ap
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import preprocessing as prep
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

class reg_metrics:
	mean_absolute_error = []
	mean_squared_error = []
	r2_score = []
	l = []
	l_es = []

class reg_params:
	learner_type = []
	cv_folds = []
	cv_grid = []
	cv_scoring = []
	num_features_input = None
 
#def compute_reg_metrics(l, l_es, i_tr, i_u):
#	rm = reg_metrics()
#	n_ts = sum(sum([((i_tr[s]==False) & i_u[s]) for s in range(len(l_es))]))
#	l_ = pd.DataFrame([0.0]*n_ts)
#	l_es_ = pd.DataFrame([0.0]*n_ts)
#	c = -1
#	for j in range(len(l_es)):
#		h = l[~i_tr[j] & i_u[j]].index
#		for i in range(len(h)):
#			c = c+1
#			l_.loc[c] = l.loc[h[i]].values
#			l_es_.loc[c] = l_es[j].loc[i].values
#
#	l_ = l_.values.flatten().astype('float')
#	l_es_ = l_es_.values.flatten().astype('float')
#
#	rm.mean_absolute_error = metrics.mean_absolute_error(l_, l_es_)
#	rm.mean_squared_error = metrics.mean_squared_error(l_, l_es_)
#	rm.r2_score = metrics.r2_score(l_, l_es_)
#	rm.l = l_
#	rm.l_es = l_es_
#	return rm


def save_results(l, l_es, i_tr, i_u, nf, runs_n, runs_cv_folds, fidout):
	n_clts = len(np.unique(l))
	rm = reg_metrics()

	for j in range(runs_n*runs_cv_folds):
		l_ = pd.DataFrame([l.loc[i] for i in l[~i_tr[j] & i_u[j//runs_cv_folds]].index]).values.flatten().astype('float')
		l_es_ = l_es[j].values.flatten().astype('float')

		for x,y in zip( l_,l_es_ ):
			print( x,y )

		#print l_, ' Questo e l'
                #print l_es_, ' Questo e l predetto'
		#

		rm.mean_absolute_error.append(metrics.mean_absolute_error(l_, l_es_))
		rm.mean_squared_error.append(metrics.mean_squared_error(l_, l_es_))
		rm.r2_score.append(metrics.r2_score(l_, l_es_))
		rm.l.append(l_)
		rm.l_es.append(l_es_)

	fidout.write('#samples\t' + str(sum(sum(i_u))//len(i_u)))
	fidout.write('#features\t' + str(nf) + '\n')
	fidout.write('\n#runs\t' + str(runs_n))
	fidout.write('\n#runs_cv_folds\t' + str(runs_cv_folds))
	fidout.write('\nmean_absolute_error\t' + str(np.mean(rm.mean_absolute_error)) + '\t' + str(np.std(rm.mean_absolute_error)))
	fidout.write('\nmean_squared_error\t' + str(np.mean(rm.mean_squared_error)) + '\t' + str(np.std(rm.mean_squared_error)))
	fidout.write('\nr2_square\t' + str(np.mean(rm.r2_score)) + '\t' + str(np.std(rm.r2_score)))	
	return rm


def plot_estimations(rm, outes):
	plt.scatter(rm.l, rm.l_es, c='k', label='estimations (r2 = %0.2f)' % rm.r2_score)
	plt.xlabel('True values')
	plt.ylabel('Estimated values')
	plt.title('')
	plt.legend(loc="lower right")
	[plt.savefig(outes+('.%s'%fmt), dpi=300) for fmt in ['png', 'svg']]


def read_params(args):
	parser = ap.ArgumentParser(description='MicroMetaAnalysis')
	arg = parser.add_argument

	arg( 'inp_f', metavar='INPUT_FILE', nargs='?', default=sys.stdin, type=str, help="the input dataset file [stdin if not present]")
	arg( 'out_f', metavar='OUTPUT_FILE', nargs='?', default=sys.stdout, type=str, help="the output dataset file [stdout if not present]")
	arg( '-z','--feature_identifier', type=str, default='k__', help="the feature identifier\n")
	arg( '-d','--define', type=str, help="define the regression problem\n")
	arg( '-t','--target', type=str, help="define the target domain\n")
	arg( '-b','--label_shuffling', action='store_true', help="label shuffling\n")

	arg( '-l','--learner_type', choices=['svm', 'rf', 'lgr'], help="the type of learner (regressor)\n")
	arg( '-f','--cv_folds', type=int, help="number of cross-validation folds\n")
	arg( '-g','--cv_grid', type=str, help="parameter grid for cross-validation\n")
	arg( '-s','--cv_scoring', type=str, help="scoring function for cross-validation\n")

	arg( '-w','--set_seed', action='store_true', help="setting seed\n")
	arg( '-r','--n_runs', default=20, type=int, help="number of runs\n")
	arg( '-p', '--runs_cv_folds', default=10, type=int, help="the number of cross-validation folds per run\n")

	arg( '-c','--rf_criterion', type=str, choices=['mse', 'mae'], default='mae')
	arg( '-mf','--max_number_features'\
	   , choices=['0.001', '0.01', '0.1', '0.2', '0.3', '0.5', '0.4', '0.6'\
	   , '100', 'auto', 'sqrt', '0.33', 'log2'], default='0.3')
	arg( '-nt','--number_of_trees', type=int, default=1000)
	arg( '-nsl','--number_sample_leaf', type=int, default=1)
        arg( '-oob','--oob_score', action='store_true', help='Enable out-of-bag choice in random forest')
        arg( '-df','--disable_features', action='store_true', help='Doesn\'t perform the features selection' +\
             'which follows std random forest (random forest)')
        arg( '-cc', '--choose_cut', type=str, default=None, \
            help='comma-separated list of numbers which will substitute the std features cuts (10,20,...,150)')
        arg( '-wc', '--weight_classes', default=None, type=float, nargs=2)

	arg( '-nc', '--ncores', type=int, default=10, help='-1 set to all the available cores.')
	arg( '-hv','--how_verbose', default=1, type=int, choices=[0,1,2])

	#arg( '-p','--p_test', default=0.1, type=float, help="dataset proportion to include in the test split\n")
	arg( '-u','--unique', type=str, help="unique samples to select\n")
	return vars(parser.parse_args())


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



#def save_results(rm, f, i_u):
#	pd.DataFrame([sum(sum(i_u))/len(i_u)], index=['#samples']).to_csv(par['out_f'], sep='\t', header=False, index=True, line_terminator='\n')
#	pd.DataFrame([len(f.iloc[0,:])], index=['#features']).to_csv(par['out_f'], sep='\t', header=False, index=True, line_terminator='\n', mode='a')	
#	pd.DataFrame([rm.mean_absolute_error], index=['mean_absolute_error']).to_csv(par['out_f'], sep='\t', header=False, index=True, line_terminator='\n', mode='a')
#	pd.DataFrame([rm.mean_squared_error], index=['mean_squared_error']).to_csv(par['out_f'], sep='\t', header=False, index=True, line_terminator='\n', mode='a')
#	pd.DataFrame([rm.r2_score], index=['r2_score']).to_csv(par['out_f'], sep='\t', header=False, index=True, line_terminator='\n', mode='a')


def set_reg_params(args):
	lp = reg_params()

	if par['learner_type']:
		lp.learner_type = par['learner_type']
	else:
		lp.learner_type = 'svm'

	if par['cv_folds']:
		lp.cv_folds = int(par['cv_folds'])
	elif lp.learner_type == 'svm':
		lp.cv_folds = 5

	if par['cv_grid']:
		lp.learner_type = par['cv_grid']
	elif lp.learner_type == 'svm':
		# lp.cv_grid = [ {'C': [2**s for s in range(-5,16,2)], 'epsilon': [0.001, 0.01, 0.1], 'kernel': ['linear']},
		##{'C': [2**s for s in range(-5,16,2)], 'gamma': [2**s for s in range(3,-15,-2)], 'epsilon': [0.001, 0.01, 0.1], 'kernel': ['rbf']} ]
		lp.cv_grid = [{'C':[1, 10, 100, 1000], 'epsilon':[0.001, 0.01, 0.1], 'kernel':['linear']}, \
			      {'C':[1, 10, 100, 1000], 'gamma':[0.001, 0.0001], 'epsilon':[0.001, 0.01, 0.1], 'kernel':['rbf']}]
	
	if par['cv_scoring']:
		lp.learner_type = par['cv_scoring']
	elif lp.learner_type == 'svm':
		lp.cv_scoring = 'mean_absolute_error'
 
	try:
		lp.num_features_input = float(par['max_number_features']) 
	except:
		lp.num_features_input = str(par['max_number_features'])
 
	#lp.num_cores = par['ncores'] if par['ncores'] > 0 else 'all'

	return lp


if __name__ == "__main__":

	par = read_params(sys.argv)
        if par['rf_max_features'] in ['0.001', '0.01', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.33','100', '1.0']:
                par['rf_max_features'] = float(par['rf_max_features'])

	f = pd.read_csv(par['inp_f'], sep='\t', header=None, index_col=0) #, dtype=unicode)
	f = f.T

	if par['out_f']:
                fidout = open(par['out_f'] + '.txt','w')
                fidoutes = par['out_f'] + '_estimations'
        else:
                fidout = sys.stdout

	if par['unique']:
		pf = pd.DataFrame([s.split(':') for s in par['unique'].split(',')])	
	if par['define']:
		l = pd.DataFrame(f[par['define']])
	else:
		l = pd.DataFrame(f.iloc[:,0])

	if par['label_shuffling']:
		np.random.shuffle(l.values)

	n_runs = par['n_runs']

	if par ['target']:
		runs_cv_folds = 1
	else:
		runs_cv_folds = par['runs_cv_folds'] ## to be added

	i_tr = pd.DataFrame(True, index=range(len(f.index)), columns=range(n_runs*runs_cv_folds))## classifier

	if par['target']:
		i_u = pd.DataFrame(True, index=range(len(f.index)), columns=range(n_runs))
		#t = pd.DataFrame([s.split(':') for s in par['target'].split(',')])		
		#for i in range(len(t)):
		#	i_tr[(f[t.iloc[i,0]].isin(t.iloc[i,1:])).tolist()] = False
	else:
		if par['unique']:
			i_u = pd.DataFrame(False, index=range(len(f.index)), columns=range(n_runs))
			meta_u = [s for s in f.columns if s in pf.iloc[0,0:].tolist()]
		else:
			i_u = pd.DataFrame(True, index=range(len(f.index)), columns=range(n_runs))

	for j in range(n_runs):
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

			#if par['set_seed']:
                        #        skf = StratifiedKFold(l.iloc[i_u.values.T[j],0], runs_cv_folds, shuffle=True, random_state=j)
                        #else:
                        #        skf = StratifiedKFold(l.iloc[i_u.values.T[j],0], runs_cv_folds, shuffle=True)
                        ###for i in range(runs_cv_folds):
                           ###     for s in np.where(skf.test_folds == i)[0]:
                              ###          i_tr[j*runs_cv_folds+i][ii_u[s]] = False

			## COMMENTED PART IS AN OLD IMPLEMENTATION;  I M COPYING THE NEW
			#rs = [s[1] for s in ShuffleSplit(len(ii_u), n_iter=1, test_size=float(par['runs_cv_folds'])/100.)]

			#for i in range(runs_cv_folds):
			#	for s in np.nditer(rs):
			#		#i_tr[j][ii_u[s]] = False
			#		i_tr[j*runs_cv_folds+i][ii_u[s]] = False


			skf = ShuffleSplit(n_splits = n_runs, test_size = runs_cv_folds/100., \
                            random_state = (j if par['set_seed'] else None))

			allowed_values = np.array(l.iloc[i_u.values.T[j], 0].values, dtype=np.float64)
			skf_split = skf.split(np.array([[0, 0] for q in range(len(allowed_values))], dtype=np.float64), allowed_values)
			test_folds = [tf[1] for tf in skf_split] # [train_index,test_index in skf_split]]

			for i in range(runs_cv_folds):
				for s in test_folds[i]:
					i_tr[j*runs_cv_folds + i][ii_u[s]] = False

			print( i_tr )

		#else:
		#	i_u = pd.DataFrame(True, index=range(len(f.index)), columns=range(n_runs))
		#	rs = [s[1] for s in ShuffleSplit(len(l), n_iter=n_runs, test_size=par['p_test'])]
		#	for j in range(n_runs):
		#		i_tr[j][rs[j]] = False

	i_tr = i_tr.values.T
	i_u = i_u.values.T

	if par['label_shuffling']:
                np.random.shuffle(l.values)

	feat = [s for s in f.columns if sum([s2 in s for s2 in par['feature_identifier'].split(':')])>0]
	if 'unclassified' in f.columns: feat.append('unclassified')
	f = f.loc[:,feat].astype('float')

        if not par['no_norm']:
                f = (f-f.min())/(f.max()-f.min())

	lp = set_reg_params(sys.argv)

	clf = []
	l_es = []
	global_time = time.time()

	for j in range(n_runs*runs_cv_folds):
		start_run_time = time.time()

                fi.append(feature_importance(feat, 1.0/len(feat)))

		#if lp.learner_type == 'svm':
		#	clf.append(GridSearchCV(SVR(C=1), lp.cv_grid, \
		#	cv=StratifiedKFold(l.iloc[i_tr[j] & i_u[j//runs_cv_folds],0], lp.cv_folds), scoring=lp.cv_scoring\
		#	).fit(f[i_tr[j] & i_u[j]].values, l[i_tr[j] & i_u[j]].values.flatten().astype('float')))

		if lp.learner_type == 'rf':
			clf.append(RandomForestRegressor(\
                                n_estimators=par['number_of_trees'], \
                                max_features=par["max_features"]\
			, min_samples_leaf=par['number_sample_leaf'], verbose=1, \
                        n_jobs=par['ncores'], \
                        criterion=par['rf_criterion']).fit(\
			f.loc[i_tr[j] & i_u[j//runs_cv_folds]].values\
			, l[i_tr[j] & i_u[j//runs_cv_folds]].values.flatten().astype('float')))

		elif lp.learner_type == 'lgr':
			clf.append(LogisticRegression(verbose=1, n_jobs=par['ncores'], \
			solver='liblinear').fit(f.loc[i_tr[j] & i_u[j//runs_cv_folds]].values\
                        , l[i_tr[j] & i_u[j//runs_cv_folds]].values.flatten().astype('float')))	

		elif lp.learner_type == 'lm':
                	clf.append()


		# p_es.append(pd.DataFrame(clf[j].predict_proba(f.loc[~i_tr[j] & i_u[j//runs_cv_folds], fi[j].feat_sel].values)))
                # l_es.append(pd.DataFrame([list(p_es[j].iloc[i,:]).index(max(p_es[j].iloc[i,:])) for i in range(len(p_es[j]))]))

                #print(f.loc[~i_tr[j] & i_u[j//runs_cv_folds]].values) 
                		

		l_es.append(pd.DataFrame(clf[-1].predict(f.loc[~i_tr[j] & i_u[j//runs_cv_folds]].values)))		
		elapsed_run_time = time.time() - start_run_time
                if par['how_verbose'] > 0:
		    print('%i run-time: %.4f sec.' %(j, float(elapsed_run_time)))

	reg_met = save_results(l, l_es, i_tr, i_u, len(feat), n_runs, runs_cv_folds, fidout)	
	#plot_estimations(reg_met, fidoutes)
	global_elapsed = time.time() - global_time
        if par['how_verbose'] > 0:
	    print ('global-time: %.4f sec.' %(global_elapsed))

        if lp.learner_type == 'rf':
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
                                    p_es_f_co = []
                                    l_es_f_co = []
                                    
                                    for j in range(runs_n*runs_cv_folds):

                                        clf_f.append(RandomForestRegressor(\
                                        n_estimators=par['number_of_trees'], \
                                        max_features=(k if par['feature_identifier'] != 'UniRef90' else (k if k<=128 else 0.3))\
                                        , min_samples_leaf=par['number_sample_leaf'], verbose=par['how_verbose'], \
                                        n_jobs=par['ncores'], class_weight='balanced'), \
                                        criterion=par['rf_criterion']).fit(\
                                        f.loc[i_tr[j] & i_u[j//runs_cv_folds], fi_f[j].feat_sel[:k] ].values\
                                        , l[i_tr[j] & i_u[j//runs_cv_folds]].values.flatten().astype('float')))

                                        l_es_f.append(pd.DataFrame(clf_f[-1].predict(f.loc[~i_tr[j] & i_u[j//runs_cv_folds], fi_f[j].feat_sel[:k]].values)))

                                    reg_met_f = save_results(l, l_es_f, i_tr, i_u, k, n_runs, runs_cv_folds, fidout)

                fi_ave = save_average_feature_importance(fi_f, feat)

	if par['out_f']:
		fidout.close()
	###
