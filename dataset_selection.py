#!/usr/bin/env python

import argparse as ap
import pandas as pd
import sys

def read_params(args):
	parser = ap.ArgumentParser(description='Select specific dataset from input dataset file')
	arg = parser.add_argument
	arg( 'inp_f', metavar='INPUT_FILE', nargs='?', default=sys.stdin, type=str, help="the input dataset file [stdin if not present]")
	arg( 'out_f', metavar='OUTPUT_FILE', nargs='?', default=None, type=str, help="the output dataset file")
	arg( '-z','--feature_identifier', type=str, default='k__', help="the feature identifier\n")
	arg( '-s','--select', type=str, help="the samples to select\n")
	arg( '-r','--remove', type=str, help="the samples to remove\n")
	arg( '-i','--include', type=str, help="the fields to include\n")
	arg( '-e','--exclude', type=str, help="the fields to exclude\n")
	arg( '-t','--tout', action='store_true', help="transpose output dataset file\n")
	return vars(parser.parse_args())

def main(args):
	par = read_params(args)

	f = pd.read_csv(par['inp_f'], sep='\t', header=None, index_col=0, dtype='unicode')
	f = f.T

	pf = pd.DataFrame()
	if par['select']:
		pf = pf.append(pd.DataFrame([s.split(':') for s in par['select'].split(',')],index=['select']*(par['select'].count(',')+1)))
	if par['remove']:
		pf = pf.append(pd.DataFrame([s.split(':') for s in par['remove'].split(',')],index=['remove']*(par['remove'].count(',')+1)))
	if par['include']:
		pf = pf.append(pd.DataFrame([s.split(':') for s in par['include'].split(',')],index=['include']*(par['include'].count(',')+1)))
	if par['exclude']:
		pf = pf.append(pd.DataFrame([s.split(':') for s in par['exclude'].split(',')],index=['exclude']*(par['exclude'].count(',')+1)))

	meta = [s for s in f.columns if sum([s2 in s for s2 in par['feature_identifier'].split(':')])==0]
	if 'unclassified' in meta: meta.remove('unclassified')
	feat = [s for s in f.columns if sum([s2 in s for s2 in par['feature_identifier'].split(':')])>0]
	if 'unclassified' in f.columns: feat.append('unclassified')

	for i in range(len(pf)):

		if pf.index[i] == 'select':
			f = f[f[pf.iloc[i,0]].isin(pf.iloc[i,1:])]

		if pf.index[i] == 'remove':
			f = f[-f[pf.iloc[i,0]].isin(pf.iloc[i,1:])]

		if pf.index[i] == 'include':
			if pf.iloc[i,0] != 'feature_level':
				meta = [s for s in meta if s in pf.iloc[i,0:].tolist()]
			else:
				feat = [s for s in feat if (pf.iloc[i,1] in s) | ('unclassified' in s) ]

		if pf.index[i] == 'exclude':
			if pf.iloc[i,0] != 'feature_level':
				if pf.iloc[i,0] == '_all_':
					meta = []
				else:
					meta = [s for s in meta if s not in pf.iloc[i,0:].tolist()]
			else:
				if pf.iloc[i,1] == '_all_':
					feat = []
				else:
					feat = [s for s in feat if pf.iloc[i,1] not in s]
		
	f=f.loc[:,meta+feat]

	f.loc[:,feat] = f.loc[:,feat].replace(to_replace='nd', value='0.0')
	f.drop(f.loc[:,feat].columns[f.loc[:,feat].max().astype('float')==f.loc[:,feat].min().astype('float')], axis=1, inplace=True)

	if par['out_f']:
		if par['tout']:
			f.to_csv(par['out_f'], sep='\t', header=True, index=False, line_terminator='\n')
		else:
			f.T.to_csv(par['out_f'], sep='\t', header=False, index=True, line_terminator='\n')

if __name__ == "__main__":
	main(sys.argv)