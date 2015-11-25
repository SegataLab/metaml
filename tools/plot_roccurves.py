#!/usr/bin/env python

import sys
import argparse as ap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class plot_par:
	dataset_name_prefix = ['ab_Quin_gut_liver_cirrhosis__d-disease__','ab_Zeller_fecal_colorectal_cancer--group__d-disease__','ab_metahit__d-disease__','ab_Chatelier_gut_obesity__d-disease__','ab_t2dmeta_long-t2dmeta_short__d-disease__','ab_WT2D__d-disease__']
	dataset_name_prefix_title = ['Cirrhosis','Colorectal','Metahit','Obesity','T2D','WT2D']
	dataset_name_suffix = [['l-svm_roc.txt','l-svm__b_roc.txt'],['l-rf_roc.txt','l-rf__b_roc.txt']]
	fig_size = [4, 8]
	plot_color = ['b','g','r','c','m','orange']
	plot_ls = ['-','--']
	plot_lw = 2
	plot_marker = ['None','None']
	plot_title = [[0.5,0.1,'SVM'],[0.5,0.1,'RF']]
	text_size = 10
	title = 'ROC curves'
	x_label = 'False positive rate'
	y_label = 'True positive rate'

def read_params(args):
	parser = ap.ArgumentParser(description='Plot ROC curves')
	arg = parser.add_argument
	arg( 'path', metavar='PATH', nargs='?', default=None, type=str, help="the path")
	arg( 'out_fig', metavar='OUT_FIGURE', nargs='?', default=None, type=str, help="the output figure file")
	return vars(parser.parse_args())

if __name__ == "__main__":
	par = read_params(sys.argv)
	plot_par = plot_par()
	nplots = len(plot_par.dataset_name_suffix)

	fig, ax = plt.subplots(nplots, sharex=True, sharey=True)

	for k in range(nplots):
		for i in range(len(plot_par.dataset_name_prefix)):
			for j in range(len(plot_par.dataset_name_suffix[k])):
				inp_f = open(par['path'] + plot_par.dataset_name_prefix[i] + plot_par.dataset_name_suffix[k][j],'r')
				f = [s.split('\t')[:-1] for s in inp_f.read().split('\n')[1:-1]]
				ax[k].plot(f[0], f[1], color=plot_par.plot_color[i], ls=plot_par.plot_ls[j], lw=plot_par.plot_lw, marker=plot_par.plot_marker[j])
				inp_f.close()

	fig.subplots_adjust(hspace=0)
	ax[-1].set_xlabel(plot_par.x_label, size=plot_par.text_size)
	ax[-1].tick_params(labelsize=plot_par.text_size, axis='x')
	for k in range(nplots):
		ax[k].set_ylabel(plot_par.y_label, size=plot_par.text_size)
		ax[k].tick_params(labelsize=plot_par.text_size, axis='y')
		ax[k].text(plot_par.plot_title[k][0], plot_par.plot_title[k][1], plot_par.plot_title[k][2], va='center', ha='center', size=plot_par.text_size+2)
	ax[-1].set_yticklabels(ax[-1].get_yticks()[:-1])
	ax[0].set_title(plot_par.title, size=plot_par.text_size+2)

	leg_col = [plt.Rectangle((0, 0), 1, 1, fc=s, linewidth=0) for s in plot_par.plot_color] + [plt.Line2D([0,1], [0,1], c='k', ls='-', lw=2)] + [plt.Line2D([0,1], [0,1], c='k', ls='--', lw=2)]
	leg_l = [s for s in plot_par.dataset_name_prefix_title] + ['True labels'] + ['Random labels']
	leg = ax[-1].legend(leg_col, leg_l, prop={'size':plot_par.text_size}, loc='center left', bbox_to_anchor=(1.02,1), numpoints=1)
	leg.get_frame().set_alpha(0)

	if plot_par.fig_size != []:
		fig.set_size_inches(plot_par.fig_size[0],plot_par.fig_size[1])

	fig.savefig(par['out_fig'], bbox_inches='tight')