# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ runner.py ]
#   Synopsis     [ Implement of two basic algorithms to perform frequent pattern mining: 1. Apriori, 2. Eclat. 
#   			   Find all itemsets with support > min_support. ]
#   Author       [ Ting-Wei Liu (Andi611) ]
#   Copyright    [ Copyleft(c), NTUEE, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import csv
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from apriori import apriori
from eclat import eclat


##################
# CONFIGURATIONS #
##################
def get_config():
	parser = argparse.ArgumentParser(description='frequent itemset mining argument parser')
	parser.add_argument('mode', type=str, choices=['apriori', 'eclat', '1', '2'], help='algorithm mode')
	parser.add_argument('--toy_data', action='store_true', help='use toy data for testing')
	parser.add_argument('--use_CUDA', action='store_true', help='run eclat with GPU to accelerate computation')
	parser.add_argument('--iterative', action='store_true', help='run eclat in iterative method, else use the recusrive method')
	parser.add_argument('--plot', action='store_true', help='Run all the values in the support list and plot runtime')
	parser.add_argument('--min_support', type=float, default=0.1, help='minimum support of the frequent itemset')
	parser.add_argument('--input_path', type=str, default='./data/data.txt', help='input data path')
	parser.add_argument('--output_path', type=str, default='./data/output.txt', help='output data path')
	args = parser.parse_args()
	if args.mode == '1': args.mode = 'apriori'
	elif args.mode == '2': args.mode = 'eclat'
	return args


#############
# READ DATA #
#############
def read_data(data_path, skip_header=False, toy_data=False):
	if toy_data:
		return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]
	data = []
	if not os.path.isfile(data_path): raise ValueError('Invalid data path.')
	with open(data_path, 'r', encoding='utf-8') as f:
		file = csv.reader(f, delimiter=' ', quotechar='\r')
		if skip_header: next(file, None)  # skip the headers
		for row in file:
			data.append(row)
	return data


#################
# RUN ALGORITHM #
#################
def run_algorithm(data, mode, support, iterative, use_CUDA):
	if mode == 'apriori':
		print('Running Apriori algorithm with %f support and data shape: ' % (support), np.shape(data))
		LK, suppotK = apriori(data, support)
		return LK, suppotK
	elif mode == 'eclat':
		print('Running eclat algorithm with %f support and data shape: ' % (support), np.shape(data))
		result = eclat(data, support, iterative, use_CUDA)
		return result
	else:
		raise NotImplementedError('Invalid algorithm mode.')


################
# WRITE RESULT #
################
def write_result(result, result_path):
	with open(result_path, 'w', encoding='big5') as file:
		file_data = csv.writer(file, delimiter=',', quotechar='\r')
		for itemset_K in result[0]:
			for itemset in itemset_K:
				output_string = ''
				for item in itemset: output_string += str(item)+' '
				output_string += '(' + str(result[1][itemset]) +  ')'
				file_data.writerow([output_string])
	print('Results have been successfully saved to: %s' % (result_path))
	return True


########
# MAIN #
########
"""
	main function that runs the two algorithms
"""
def main():
	args = get_config()
	data = read_data(args.input_path, toy_data=args.toy_data)
	
	if not args.plot: support_list = [args.min_support]
	elif args.mode == 'apriori': support_list = [0.35, 0.3, 0.25, 0.2]
	elif args.mode == 'eclat': support_list = [0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]
	
	duration = []
	for s in support_list:
		print('-'*77)
		start_time = time.time()
		result = run_algorithm(data, args.mode, s, args.iterative, args.use_CUDA)
		"""
			result has len()==2, 
			result[0]: the 3-dimensional Large K itemset,
			result[1]: the dictionary storing the support of each itemset
		"""
		duration.append(time.time() - start_time)
		print("Time duration: %.5f" % (duration[-1]))

	if args.plot:
		fig = plt.figure()
		ax = fig.add_subplot(111)
		plt.plot(support_list, duration, 'r')
		for xy in zip(support_list, duration):
			ax.annotate('(%s, %.5s)' % xy, xy=xy, textcoords='data')
		plt.ylabel('execution time (seconds)')
		plt.xlabel('minimum support')
		plt.title(args.mode)
		plt.grid()
		fig.savefig('./data/' + args.mode + '_plot.jpeg')
	else:
		done = write_result(result, args.output_path)


if __name__ == '__main__':
	main()

