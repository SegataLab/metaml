#!/usr/bin/env python

import argparse as ap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

class plot_par:
	aspect_ratio = 'auto'
	bbox = dict(boxstyle='round4, pad=0.3', fc='none', ec='k', lw=1)
	bbox_plot = 'yes'
	cbar_extend = 'min'
	cbar_shrink = 0.9
	cbar_ticksnumber = 6
	cbar_ylabel = 'AUC'
	cmap_color = 'hot'
	cmap_colorbad = ['w', 1]
	cmap_vmin = 0.5
	cmap_vmax = 1.0
	fig_size = []
	lin_c = ['k','k']
	lin_ls = ['-','--']
	lin_lw = [2,0.5]
	text_col = 'k'
	text_col2 = 'w'
	text_size = 8
	text_weight = 1000
	text_weight_plot = 'yes'
	text_white_min = 0
	text_white_max = 0.65
	ticks_spacing = 0.05
	title = ''
	xticks_n = 1
	yticks_n = 1

def isfloat(value):
	try:
		float(value)
    		return True
  	except ValueError:
    		return False

def read_params(args):
	parser = ap.ArgumentParser(description='Plot annotated heatmap')
	arg = parser.add_argument
	arg( 'inp_f', metavar='INPUT_FILE', nargs='?', default=sys.stdin, type=str, help="the input results file [stdin if not present]")
	arg( 'out_fig', metavar='OUT_FIGURE', nargs='?', default=None, type=str, help="the output figure file")
	return vars(parser.parse_args())

if __name__ == "__main__":
	par = read_params(sys.argv)
	plot_par = plot_par()

	f = pd.read_csv(par['inp_f'], sep='\t', header=None, index_col=None, dtype=unicode)

	fm = pd.DataFrame(data=np.nan, index=range(f.shape[0]-plot_par.xticks_n), columns=range(f.shape[1]-plot_par.yticks_n))
	for i in range(f.shape[0]-plot_par.xticks_n):
		for j in range(plot_par.yticks_n,f.shape[1]):
			t = f.iloc[i,j].split('\\n')[0]
			if isfloat(t):
				if float(t)<=1:
					fm.iloc[i,j-plot_par.yticks_n] = float(t)

	cmap = plt.get_cmap(plot_par.cmap_color)
	cmap.set_bad(plot_par.cmap_colorbad[0], plot_par.cmap_colorbad[1])

	fig, ax = plt.subplots()
	cax = ax.imshow(fm, cmap=cmap, interpolation='nearest', vmin=plot_par.cmap_vmin, vmax=plot_par.cmap_vmax)

	for i in range(f.shape[0]-plot_par.xticks_n):
		for j in range(plot_par.yticks_n,f.shape[1]):
			if f.iloc[i,j] != '*empty':
				check = 'nor'
				col = plot_par.text_col
				if isfloat(fm.iloc[i,j-plot_par.yticks_n]):
					if plot_par.yticks_n == 1:
						max_tot = fm[j-plot_par.yticks_n].max()
						max_par = (fm[j-plot_par.yticks_n][f.iloc[:-plot_par.xticks_n,0] == f.iloc[i,0]]).max()
					elif plot_par.yticks_n == 2:
						max_tot = (fm[j-plot_par.yticks_n][f.iloc[:-plot_par.xticks_n,1] == f.iloc[i,1]]).max()
						max_par = (fm[j-plot_par.yticks_n][(f.iloc[:-plot_par.xticks_n,0] == f.iloc[i,0]) & (f.iloc[:-plot_par.xticks_n,1] == f.iloc[i,1])]).max()
					if fm.iloc[i,j-plot_par.yticks_n] == max_tot:
						check = 'tot'
					elif fm.iloc[i,j-plot_par.yticks_n] == max_par:
						check = 'par'
					if (fm.iloc[i,j-plot_par.yticks_n]>plot_par.text_white_min) & (fm.iloc[i,j-plot_par.yticks_n]<plot_par.text_white_max):
						col = plot_par.text_col2
				if (check == 'tot') & (plot_par.bbox_plot=='yes'):
					ax.text(j-plot_par.yticks_n, i, '\n'.join(f.iloc[i,j].split('\\n')), va='center', ha='center', size=plot_par.text_size, color=col, weight=plot_par.text_weight, bbox=plot_par.bbox)
				elif (check == 'par') & (plot_par.text_weight_plot=='yes'):
					ax.text(j-plot_par.yticks_n, i, '\n'.join(f.iloc[i,j].split('\\n')), va='center', ha='center', size=plot_par.text_size, color=col, weight=plot_par.text_weight)
				else:
					ax.text(j-plot_par.yticks_n, i, '\n'.join(f.iloc[i,j].split('\\n')), va='center', ha='center', size=plot_par.text_size, color=col)

	for i in range(f.shape[0]-2):
		if f.iloc[i,0] != f.iloc[i+1,0]:
				plt.plot([-0.5, f.shape[1]-plot_par.yticks_n-0.5], [i+0.5, i+0.5], c=plot_par.lin_c[0], ls=plot_par.lin_ls[0], lw=plot_par.lin_lw[0])
		if plot_par.yticks_n == 2:
			if f.iloc[i,1] != f.iloc[i+1,1]:
				plt.plot([-0.5, f.shape[1]-plot_par.yticks_n-0.5], [i+0.5, i+0.5], c=plot_par.lin_c[1], ls=plot_par.lin_ls[1], lw=plot_par.lin_lw[1])			

	xticks = [[s-plot_par.yticks_n for s in range(plot_par.yticks_n,f.shape[1]) if f.iloc[-1,s] != '*empty'] for s2 in range(plot_par.xticks_n)]
	xticklabels = [[f.iloc[-1-s2,s+plot_par.yticks_n] for s in xticks[s2]] for s2 in range(plot_par.xticks_n)]
	xticklabels_a = []
	for i in range(len(xticklabels)):
		for j in range(len(xticklabels[0])):
			t = ''
			for k in range(i+1):
				t = t + '__' + xticklabels[k][j]
			xticklabels_a.append(t)
	xticklabels_au = np.unique(xticklabels_a)
	xticks_u = []
	xticks_us = []
	xticklabels_u = []
	for k in xticklabels_au:
		xticklabels_u.append(k.split('__')[-1])
		xticks_u.append(0.0)
		xticks_us.append(0.0)
		c = 0
		count = 0
		for i in range(len(xticklabels)):
			for j in range(len(xticklabels[0])):
				if xticklabels_a[c]==k:
					count = count+1
					xticks_u[-1] = xticks_u[-1]+xticks[i][j]
					xticks_us[-1] = xticks_us[-1]+i
				c = c+1
		xticks_u[-1] = xticks_u[-1]/count
		xticks_us[-1] = xticks_us[-1]/count
	ax.xaxis.tick_top()
	ax.tick_params(bottom='off', right='off')
	ax.set_xticks(xticks_u)
	ax.set_xticklabels(xticklabels_u, size=plot_par.text_size)
	for i, j in zip(ax.get_xticklabels(), xticks_us):
		i.set_y(1+plot_par.ticks_spacing*(plot_par.xticks_n-j-1))		

	yticks = [[s for s in range(f.shape[0]-plot_par.xticks_n) if f.iloc[s,0] != '*empty'] for s2 in range(plot_par.yticks_n)]
	yticklabels = [[f.iloc[s,s2] for s in yticks[s2]] for s2 in range(plot_par.yticks_n)]
	yticklabels_a = []
	for i in range(len(yticklabels)):
		for j in range(len(yticklabels[0])):
			t = ''
			for k in range(i+1):
				t = t + '__' + yticklabels[k][j]
			yticklabels_a.append(t)
	yticklabels_au = np.unique(yticklabels_a)
	yticks_u = []
	yticks_us = []
	yticklabels_u = []
	for k in yticklabels_au:
		yticklabels_u.append(k.split('__')[-1])
		yticks_u.append(0.0)
		yticks_us.append(0.0)
		c = 0
		count = 0
		for i in range(len(yticklabels)):
			for j in range(len(yticklabels[0])):
				if yticklabels_a[c]==k:
					count = count+1
					yticks_u[-1] = yticks_u[-1]+yticks[i][j]
					yticks_us[-1] = yticks_us[-1]+i
				c = c+1
		yticks_u[-1] = yticks_u[-1]/count
		yticks_us[-1] = yticks_us[-1]/count
	ax.set_yticks(yticks_u)
	ax.set_yticklabels(yticklabels_u, size=plot_par.text_size, rotation='vertical')
	for i, j in zip(ax.get_yticklabels(), yticks_us):
		i.set_x(plot_par.ticks_spacing*(-plot_par.yticks_n+j+1))

	ax.set_xlim(-0.5, f.shape[1]-plot_par.yticks_n-0.5)
	ax.set_ylim(f.shape[0]-plot_par.xticks_n-0.5, -0.5)
	ax.set_aspect(plot_par.aspect_ratio)

	cbar = fig.colorbar(cax, shrink=plot_par.cbar_shrink, extend=plot_par.cbar_extend, ticks=[plot_par.cmap_vmin+s*(plot_par.cmap_vmax-plot_par.cmap_vmin)/(plot_par.cbar_ticksnumber-1) for s in range(plot_par.cbar_ticksnumber)])
	cbar.ax.tick_params(labelsize=plot_par.text_size)
	cbar.ax.set_ylabel(plot_par.cbar_ylabel, size=plot_par.text_size)

	if plot_par.title != []:
		ax.text(float(fm.shape[1]-1)/2, -len(xticks)-1, plot_par.title, va='center', ha='center', size=plot_par.text_size+2)

	if plot_par.fig_size != []:
		fig.set_size_inches(plot_par.fig_size[0],plot_par.fig_size[1])

	if par['out_fig']:
		fig.savefig(par['out_fig'], bbox_inches='tight')
	plt.show()