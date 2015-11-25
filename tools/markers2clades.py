#!/usr/bin/env python

import os
import sys
import argparse as ap
import pandas as pd
from scipy.stats import mode

def read_params(args):
	parser = ap.ArgumentParser(description='Markers to clades for MicroMetaAnalysis output file')
	arg = parser.add_argument
	arg( 'inp_f', metavar='INPUT_FILE', nargs='?', default=sys.stdin, type=str, help="the input file [stdin if not present]")
	arg( 'inp_m2c_f', metavar='INPUT_M2C_FILE', nargs='?', default=None, type=str, help="the input markers to clades file")
	arg( 'out_f', metavar='OUTPUT_FILE', nargs='?', default=None, type=str, help="the output file [stdout if not present]")
	return vars(parser.parse_args())

if __name__ == '__main__':
	par = read_params(sys.argv)

	inp_f = open(par['inp_f'],'r')
	f = [s.split('\t') for s in ((inp_f.read()).split('Feature importance and abundance')[-1]).split('\n')[1:-1]]
	inp_f.close()
	m2c = (pd.read_csv(par['inp_m2c_f'], sep='\t', header=None, index_col=0, dtype=unicode)).T

	if par['out_f']:
		fidout = open(par['out_f'],'w')
	else:
		fidout = sys.stdout

	for t in range(len(f)):
		fidout.write(str(t) + '\t' + f[t][1] + '\t' + str(mode(m2c[f[t][1]].T.values)[0])[3:-2] + '\t' + f[t][-2] + '\n')

	if par['out_f']:
		fidout.close()