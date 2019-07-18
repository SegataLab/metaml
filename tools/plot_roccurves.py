#!/usr/bin/env python

import argparse as ap
import itertools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy import interpolate
from sklearn import metrics

class plot_par:
	dataset_name_prefix = [ 'WirbelJ_2018_HMP2_Ulcerative_Colitis'
			,	'WirbelJ_2018_HMP2_Crohn_Disease'
			,	'WirbelJ_2018_HMP2_Controls'
			,	'WirbelJ_2018_Metahit_Ulcerative_Colitis'
			,	'WirbelJ_2018_Metahit_Crohn_Disease'
			,	'WirbelJ_2018_Metahit_Controls' ]

		#'abundance_cirrhosis__d-disease__'
		#,'abundance_colorectal--group__d-disease__'
		#,'abundance_ibd__d-disease__'
		#,'abundance_obesity__d-disease__'
		#,'abundance_t2d_long-t2d_short__d-disease__'
		#,'abundance_WT2D__d-disease__']


	dataset_name_prefix_title = [ 'UC(HMP2)', 'CD(HMP2)', 'CTRL(HMP2)', 'UC(MH)', 'CD(MH)', 'CTRL(MH)' ]
	dataset_name_suffix = [ [ '_estimations.txt', '_estimations.txt', '_estimations.txt', '_estimations.txt', '_estimations.txt', '_estimations.txt' ] ]

	#,['l-svm_estimations.txt','l-svm__b_estimations.txt']]

	factor = 2.26
	fig_size = [4, 8] if len(dataset_name_suffix)==1 else [4,4]
	plot_alpha = 0.2
	plot_color = ['b','g','r','c','m','orange','goldenrod']
	plot_ls = ['-','--']
	plot_lw = 2
	plot_marker = ['None','None']
	plot_title = [[0.5,0.1,'Validation Cohort + GI tract Samples']]  #,[0.5,0.1,'SVM']]
	text_size = 10
	title = 'ROC curves'
	x_label = 'False positive rate'
	y_label = 'True positive rate'


def read_params(args):
	parser = ap.ArgumentParser(description='Plot ROC curves')
	arg = parser.add_argument
	arg( 'path', metavar='PATH', nargs='?', default=None, type=str, help="the path")
	arg( 'out_fig', metavar='OUT_FIGURE', nargs='?', default=None, type=str, help="the output figure file")
	arg( '-p','--runs_cv_folds', default=10, type=int, help="the number of cross-validation folds per run\n")
	return vars(parser.parse_args())

if __name__ == "__main__":
	par = read_params(sys.argv)
	plot_par = plot_par()
	nplots = len(plot_par.dataset_name_suffix)

	fig, ax = plt.subplots(nplots, sharex=True, sharey=True)
        if nplots == 1:
            ax = [ ax ]

	for k in range(nplots):
		for i in range(len(plot_par.dataset_name_prefix)):
			for j in range(len(plot_par.dataset_name_suffix[k])):
				f = open(par['path'] + plot_par.dataset_name_prefix[i] + plot_par.dataset_name_suffix[k][j],'r').read().split("#features")[1]
				l_ = [map(int,t.split('\n')[0].split('\t')[1:-1]) for t in f.split('true labels')[1:]]
				p_es_pos_ = [map(float,t.split('\n')[0].split('\t')[1:-1]) for t in f.split('estimated probabilities')[1:]]

				fpr_all, tpr_all, thresholds_all = metrics.roc_curve(list(itertools.chain(*l_)), list(itertools.chain(*p_es_pos_)))

				tpr_i = []
				for s in range(len(l_)):
					fpr, tpr, thresholds = metrics.roc_curve(l_[s], p_es_pos_[s])
					if not np.isnan(tpr[0]):
						tpr_i.append(interpolate.interp1d(fpr, tpr, 'nearest')(fpr_all))

                                print ax
                                print len(plot_par.plot_color), ' lunghezza colori: ', i, ' ( indice colori )'
                                print plot_par.plot_ls, ' par plot ls, la sua lunghezza:  ', len(plot_par.plot_ls), j, ' la cazzo di j.. ' 


				ax[k].fill_between(fpr_all, tpr_all-np.std(tpr_i, axis=0)*plot_par.factor/np.sqrt(par['runs_cv_folds']), tpr_all+np.std(tpr_i, axis=0)*plot_par.factor/np.sqrt(par['runs_cv_folds']), color=plot_par.plot_color[i], lw=0, alpha=plot_par.plot_alpha)
				ax[k].plot(fpr_all, tpr_all, color=plot_par.plot_color[i], ls='-', lw=plot_par.plot_lw, marker=plot_par.plot_marker[0])

				#plot_par.plot_ls[j], lw=plot_par.plot_lw, marker=plot_par.plot_marker[j])

	fig.subplots_adjust(hspace=0)
	ax[-1].set_xlabel(plot_par.x_label, size=plot_par.text_size)
	ax[-1].tick_params(labelsize=plot_par.text_size, axis='x')
	for k in range(nplots):
		ax[k].set_ylabel(plot_par.y_label, size=plot_par.text_size)
		ax[k].tick_params(labelsize=plot_par.text_size, axis='y')
		ax[k].text(plot_par.plot_title[k][0], plot_par.plot_title[k][1], plot_par.plot_title[k][2], va='center', ha='center', size=plot_par.text_size+2)
		ax[k].set_xlim([0.0, 1.0])
		ax[k].set_ylim([0.0, 1.0])
	ax[-1].set_yticklabels(ax[-1].get_yticks()[:-1])
	ax[0].set_title(plot_par.title, size=plot_par.text_size+2)

	leg_col = [plt.Rectangle((0, 0), 1, 1, fc=s, linewidth=0) for s in plot_par.plot_color] + [plt.Line2D([0,1], [0,1], c='k', ls='-', lw=2)] + [plt.Line2D([0,1], [0,1], c='k', ls='--', lw=2)]
	leg_l = [s for s in plot_par.dataset_name_prefix_title] + ['True labels'] + ['Shuffled labels']
	leg = ax[-1].legend(leg_col, leg_l, prop={'size':plot_par.text_size}, loc='center left', bbox_to_anchor=(1.02,1), numpoints=1)
	leg.get_frame().set_alpha(0)

	if plot_par.fig_size != []:
		fig.set_size_inches(plot_par.fig_size[0],plot_par.fig_size[1])

	fig.savefig(par['out_fig'], bbox_inches='tight')
