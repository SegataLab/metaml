#!/usr/bin/env python

import argparse as ap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from scipy.stats import mode

class plot_par:
	bar_color = ['royalblue','green','crimson']
	bar_edgecolor = 'none'
	bar_stacked = 'no'
	bar_width = 0.48
	cmap_color = 'jet'
	feature_identifier = ['k__','GeneID:gi|']
	feature_number = [25, 25]
	feature_type = [0, 1]
	n_species_togroup = 5
	text_size = 10
	text_size_y = 10
	text_style = 'italic'
	x_label_s = ['Relative importance (in blue) [%]', 'Relative importance [%]']
	x_label_a = ['Healthy (in green) and diseased (in red)\n average relative abundance [%]', 'Healthy (in green) and diseased (in red)\n average presence [%]']
	y_label = ['Markers', 12]
	xticks_n = 5

def read_params(args):
	parser = ap.ArgumentParser(description='Plot barchart')
	arg = parser.add_argument
	arg( 'inp_f', metavar='INPUT_FILE', nargs='?', default=None, type=str, help="the input file")
	arg( 'inp_data_f', metavar='INPUT_DATA_FILE', nargs='?', default=None, type=str, help="the input data file")
	arg( 'out_fig', metavar='OUT_FIGURE', nargs='?', default=None, type=str, help="the output figure file")
	arg( 'inp_m2c_f', metavar='INPUT_M2C_FILE', nargs='?', default=None, type=str, help="the input markers to clades file")
	arg( '-d','--define', type=str, help="define the classification problem\n")
	arg( '-t','--title', type=str, default='', help="title\n")
	arg( '-s','--scores', action='store_true', help="plot scores only\n")
	return vars(parser.parse_args())

if __name__ == "__main__":
	par = read_params(sys.argv)
	plot_par = plot_par()
	nplots = len(par['inp_f'].split(':'))

	if par['scores']:	
		fig, ax = plt.subplots(1, nplots)
		if nplots == 1:
			ax = [ax]
	else:
		fig, ax2 = plt.subplots(1, nplots)
		if nplots == 1:
			ax2 = [ax2]
		ax = [s.twiny() for s in ax2]

	for inp_ft in range(nplots):
		inp_f = open(par['inp_f'].split(':')[inp_ft],'r')
		f = [s.split('\t') for s in ((inp_f.read()).split('Feature importance (ranking, name, average, std)')[-1]).split('\n')[1:-1]]
		inp_f.close()

		fdata = pd.read_csv(par['inp_data_f'].split(':')[inp_ft], sep='\t', header=None, index_col=0, dtype=unicode).T
		feat = [s for s in fdata.columns if sum([s2 in s for s2 in plot_par.feature_identifier[inp_ft].split(':')])>0]
		n_samples = len(fdata)
		l = pd.DataFrame([0]*n_samples)
		if par['define']:			
			d = pd.DataFrame([s.split(':') for s in par['define'].split(',')])				
			for i in range(len(d)):
				l[(fdata[d.iloc[i,1]].isin(d.iloc[i,2:])).tolist()] = d.iloc[i,0]
		fdata = fdata.loc[:,feat]	

		f_ind = np.arange(plot_par.feature_number[inp_ft])
		f_text = [f[s][1] for s in range(plot_par.feature_number[inp_ft])]
		f_s = [100*float(f[s][2]) for s in range(plot_par.feature_number[inp_ft])]
		f_ssd = [100*float(f[s][3]) for s in range(plot_par.feature_number[inp_ft])]		
		if par['scores'] == False:
			f_a = []
			for t in np.unique(l):
				if plot_par.feature_type[inp_ft]==1:
					f_a.append([100*np.mean(fdata[s].loc[[s2+1 for s2 in range(len(l)) if l.values[s2]==t]].astype('float')) for s in f_text])
				else:
					f_a.append([np.mean(fdata[s].loc[[s2+1 for s2 in range(len(l)) if l.values[s2]==t]].astype('float')) for s in f_text])
					#f_a.append([100*np.mean(fdata[s].loc[[s2+1 for s2 in range(len(l)) if l.values[s2]==t]].astype('float')) for s in f_text])

		if plot_par.feature_type[inp_ft]==1:	
			m2c = (pd.read_csv(par['inp_m2c_f'], sep='\t', header=None, index_col=0, dtype=unicode)).T
			f_text_s = [str(mode(m2c[s].T.values)[0])[3:-2] for s in f_text]
			f_text_su = [f_text_s[s] for s in sorted(np.unique(f_text_s, return_index=True)[1])[:plot_par.n_species_togroup]] + ['Other']
			b = [s for s in range(plot_par.feature_number[inp_ft]) if (f_text_s[s] in f_text_su)]
			f_text_sl = [1.0]*plot_par.feature_number[inp_ft]
			for s in b:
				f_text_sl[s] = float(f_text_su.index(f_text_s[s]))/plot_par.n_species_togroup
			for s in range(len(f_text_su)):
				if 's__' in f_text_su[s]:
					f_text_su[s] = ' '.join(f_text_su[s].split('s__')[-1].split('|t__')[0].split('_'))
				elif 'g__' in f_text_su[s]:
					f_text_su[s] = 'g: ' + ' '.join(f_text_su[s].split('g__')[-1].split('_'))
				elif 'f__' in f_text_su[s]:
					f_text_su[s] = 'f: ' + ' '.join(f_text_su[s].split('f__')[-1].split('_'))
		else:
			for s in range(len(f_text)):
				if 's__' in f_text[s]:
					f_text[s] = ' '.join(f_text[s].split('s__')[-1].split('|t__')[0].split('_'))
				elif 'g__' in f_text[s]:
					f_text[s] = 'g: ' + ' '.join(f_text[s].split('g__')[-1].split('_'))
				elif 'f__' in f_text[s]:
					f_text[s] = 'f: ' + ' '.join(f_text[s].split('f__')[-1].split('_'))
				if 'unclassified' in f_text[s]:
					f_text[s] = f_text[s][:-12] + 'spp.'

		if plot_par.feature_type[inp_ft]==1:
			cmap = plt.get_cmap(plot_par.cmap_color)
			for s in range(plot_par.feature_number[inp_ft]):
				ax[inp_ft].barh(f_ind[s], f_s[s], xerr=[[0], [f_ssd[s]]], height=plot_par.bar_width, color=cmap(f_text_sl[s]), ecolor=cmap(f_text_sl[s]), edgecolor=plot_par.bar_edgecolor)
		else:
			ax[inp_ft].barh(f_ind, f_s, xerr=[[0]*len(f_ssd), f_ssd], height=plot_par.bar_width, color=plot_par.bar_color[0], ecolor=plot_par.bar_color[0], edgecolor=plot_par.bar_edgecolor)
		if par['scores'] == False:
			for s in range(plot_par.feature_number[inp_ft]):
				if plot_par.bar_stacked == 'yes':
					for t in np.argsort([-f_a[t][s] for t in range(len(f_a))]):
						ax2[inp_ft].barh(f_ind[s]+plot_par.bar_width, f_a[t][s], height=plot_par.bar_width/2, color=plot_par.bar_color[t+1], edgecolor=plot_par.bar_edgecolor)
				else:
					for t in range(len(f_a)):
						ax2[inp_ft].barh(f_ind[s]+plot_par.bar_width*(1+float(t)/len(f_a)), f_a[t][s], height=plot_par.bar_width/len(f_a), color=plot_par.bar_color[t+1], edgecolor=plot_par.bar_edgecolor)					

		if plot_par.feature_type[inp_ft]==1:
			ax[inp_ft].get_yaxis().set_visible(False)
			if par['scores'] == False:
				ax2[inp_ft].get_yaxis().set_visible(False)
		else:
			ax[inp_ft].set_yticks(f_ind+0.5)
			ax[inp_ft].set_yticklabels(f_text)
			[s.set_style(plot_par.text_style) for s in ax[inp_ft].yaxis.get_ticklabels()]
			ax[inp_ft].tick_params(labelsize=plot_par.text_size_y, axis='y')
			if par['scores'] == False:
				ax2[inp_ft].set_yticks(f_ind+0.5)
				ax2[inp_ft].set_yticklabels(f_text)
				[s.set_style(plot_par.text_style) for s in ax2[inp_ft].yaxis.get_ticklabels()]
				ax2[inp_ft].tick_params(labelsize=plot_par.text_size_y, axis='y')
		ax[inp_ft].tick_params(left='off', right='off')
		if par['scores'] == False:
		 	ax2[inp_ft].tick_params(left='off', right='off')

		ax[inp_ft].set_xlabel(plot_par.x_label_s[inp_ft], size=plot_par.text_size)
		ax[inp_ft].tick_params(labelsize=plot_par.text_size, axis='x')
		ax[inp_ft].locator_params(nbins=plot_par.xticks_n, axis='x')
		if par['scores']:
			ax[inp_ft].tick_params(top='off')
		else:
			ax2[inp_ft].set_xlabel(plot_par.x_label_a[inp_ft], size=plot_par.text_size)		
			ax2[inp_ft].tick_params(labelsize=plot_par.text_size, axis='x')
			ax2[inp_ft].locator_params(nbins=plot_par.xticks_n, axis='x')

		if plot_par.feature_type[inp_ft]==1:
			leg_col = [plt.Rectangle((0, 0), 1, 1, fc=cmap(float(s)/plot_par.n_species_togroup), linewidth=0) for s in range(plot_par.n_species_togroup+1)]
			leg_l = [' '.join(s.split('_')) for s in f_text_su]
			leg = plt.legend(leg_col, leg_l, prop={'size':plot_par.text_size, 'style':plot_par.text_style}, loc='center left', bbox_to_anchor=(1.02,0.5))
			leg.get_frame().set_alpha(0)

			ax[inp_ft].text(-0.02, plot_par.feature_number[inp_ft]/2, plot_par.y_label[0], horizontalalignment='center', verticalalignment='center', rotation=90, fontsize=plot_par.y_label[1])

		ax[inp_ft].invert_yaxis()
		ax2[inp_ft].set_xscale('log')
	
	fig.subplots_adjust(wspace=0.15)
	fig.suptitle(par['title'], size=plot_par.text_size+2, y=1)

	if par['out_fig']:
		fig.savefig(par['out_fig'], bbox_inches='tight')