#!/usr/bin/env python

import argparse as ap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

class plot_par:
	feature_identifier = 'k__'
	scatter_alpha = 0.5
	scatter_size = 50
	x_label = 'Average relative abundance [%]'
	y_label = 'Average relative importance [%]'

def read_params(args):
	parser = ap.ArgumentParser(description='Plot correlation between abundance and feature importance')
	arg = parser.add_argument
	arg( 'inp_f', metavar='INPUT_FILE', nargs='?', default=None, type=str, help="the input file")
	arg( 'inp_data_f', metavar='INPUT_DATA_FILE', nargs='?', default=None, type=str, help="the input data file")
	arg( 'out_fig', metavar='OUT_FIGURE', nargs='?', default=None, type=str, help="the output figure file")
	arg( '-t','--title', type=str, default='', help="title\n")
	return vars(parser.parse_args())

if __name__ == "__main__":
	par = read_params(sys.argv)
	plot_par = plot_par()

	inp_f = open(par['inp_f'],'r')
	f = [s.split('\t') for s in ((inp_f.read()).split('Feature importance (ranking, name, average, std)')[-1]).split('\n')[1:-1]]
	inp_f.close()

	fdata = pd.read_csv(par['inp_data_f'], sep='\t', header=None, index_col=0, dtype=unicode).T
	feat = [s for s in fdata.columns if sum([s2 in s for s2 in plot_par.feature_identifier.split(':')])>0]
	fdata = fdata.loc[:,feat]
	feat_n = len(feat)

	f_text = [f[s][1] for s in range(feat_n)]
	f_a = [np.mean(fdata[s].astype('float')) for s in f_text]
	f_s = [100*float(f[s][2]) for s in range(feat_n)]

	fig, ax = plt.subplots()
	plt.scatter(f_a, f_s, s=plot_par.scatter_size, edgecolors='None', alpha=plot_par.scatter_alpha)

	fig.suptitle(par['title'] + ' (corr=' + '%.2f' % np.corrcoef(f_a, f_s)[0,1] + ')')
	ax.set_xlabel(plot_par.x_label)
	ax.set_ylabel(plot_par.y_label)
	ax.set_xlim([np.min(f_a), np.max(f_a)*1.05])
	ax.set_ylim([np.min(f_s), np.max(f_s)*1.05])

	if par['out_fig']:
		fig.savefig(par['out_fig'], bbox_inches='tight')
	plt.show()