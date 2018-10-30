# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ eclat.py ]
#   Synopsis     [ Implement of the Eclat algorithm with Vertical bitvector data ]
#   Author       [ Ting-Wei Liu (Andi611) ]
#   Copyright    [ Copyleft(c), NTUEE, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import numpy as np
try:
	import pycuda.gpuarray as gpuarray
	import pycuda.driver as cuda
	import pycuda.autoinit
	from pycuda.compiler import SourceModule
	CUDA_FLAG = True
except:
	CUDA_FLAG = False

###################################
# COMPUTE VERTICAL BITVECTOR DATA #
###################################
"""
	- Transform the original horizontal transaction data into vertical bitvector​ data representation
	- Input: 
		- data: M x N matrix with M = number of transactions, N = number of items in each transaction
	- Output:
		- vb_data: a vertical bitvector with shape [num_trans, num_transactions], 
				   in each row of a distinct item, if the item appears in transaction i, then the i-th index in that row is 1
		- idx2item: idx to item dictionary mapping
"""
def compute_vertical_bitvector_data(data):
	#---build item to idx mapping---#
	idx = 0
	item2idx = {}
	for transaction in data:
		for item in transaction:
			if not item in item2idx:
				item2idx[item] = idx
				idx += 1
	idx2item = { idx : str(int(item)) for item, idx in item2idx.items() }
	#---build vertical data---#
	vb_data = np.zeros((len(item2idx), len(data)), dtype=int)
	for trans_id, transaction in enumerate(data):
		for item in transaction:
			vb_data[item2idx[item], trans_id] = 1
	print('Data transformed into vertical bitvector representation with shape: ', np.shape(vb_data))
	return vb_data, idx2item


##############
# COMPUTE L1 #
##############
"""
	- Derivation of large 1-itemsets L1: 
		- At the first iteration, check its support by scanning all the items and sum of its bit vector.
	- Input: 
		- data: vertical bitvector with shape [num_trans, num_transactions]
		- num_trans: number of transactions
		- min_support: minimum support
	- Output:
		- L1: a list of frozensets containing the Large 1 set L1
		- support_list: a dictionary recording the supports for each itemsets in L1
"""
def compute_L1(data, idx2item, num_trans, min_support):
	L1 = []
	support_list = {}
	for idx, bit_list in enumerate(data):
		support = np.sum(bit_list) / num_trans
		if support >= min_support:
			support_list[frozenset([idx2item[idx]])] = bit_list
			L1.append([idx2item[idx]])
	return list(map(frozenset, sorted(L1))), support_list


##############
# COMPUTE CK #
##############
"""
	- Performes a scan in LK-1 and compute the next large itemset LK,
	  At the k-th iteration, the LK set are those whose every (k-1) item subset is in Lk-1,
	  and have a sum of the bit vector union greater than minimum support.
	  (k-1) is inclusive, so in implementation we need to set index to [:k-2]
	- Input:
		- LK_: the large K-1 itemset, LK-1
		- k: the k-th iteration
		- support_list: a dictionary recording the supports of all L sets
		- num_trans: number of transactions
		- min_support: minimum support
	- Output:
		- LK: the large itemset LK for the k-th iteration
		- supportK: a dictionary recording the supports for each itemsets in LK
"""
def compute_LK(LK_, support_list, k, num_trans, min_support):
	LK = []
	supportK = {}
	for i in range(len(LK_)):
		for j in range(i+1, len(LK_)):  # enumerate all combinations in the Lk-1 itemsets
			L1 = sorted(list(LK_[i])[:k-2])
			L2 = sorted(list(LK_[j])[:k-2])
			if L1 == L2: # if the first k-1 terms are the same in two itemsets, calculate the intersection support
				union = np.multiply(support_list[LK_[i]], support_list[LK_[j]])
				union_support = np.sum(union) / num_trans
				if union_support >= min_support:
					new_itemset = frozenset(sorted(list(LK_[i] | LK_[j])))
					if new_itemset not in LK:
						LK.append(new_itemset)
						supportK[new_itemset] = union
	return sorted(LK), supportK


################
# ECLAT RUNNER #
################
"""
	The recursive eclat runner that runs the eclat algorithm recursively.
	- Usage: recursively call on the run() function.
	- Input:
		- prefix: an empty list
		- supportK: a list containing pairs of (item, bit vector) converted from vertical bitvector
	- Output:
		- support_list: an unsorted dictionary recording the supports found by the recursive method.
"""
class eclat_runner:

	def __init__(self, num_trans, min_support, use_CUDA):
		self.num_trans = num_trans
		self.min_support = min_support
		self.support_list = {}
		self.use_CUDA = use_CUDA
		if self.use_CUDA:
			self.block = (256, 1, 1)
			dx, mx = divmod(self.num_trans, self.block[0])
			dy, my = divmod(0, self.block[1])
			# self.grid = (int((dx + (mx>0)) * self.block[0]), int((dy + (my>0)) * self.block[1]))
			self.grid = (1, 1)
			print("Accelerating Eclat computation with GPU!")
			print("Using Block =", self.block)
			print("Using Grid =", self.grid)
		else:
			print("Failed to import Pycuda! Machine does not support GPU computation.")


	def run(self, prefix, supportK):
		print('Running Eclat in recursive: number of itemsets found:', len(self.support_list), end='\r')
		while supportK:
			itemset, bitvector = supportK.pop(0)
			support = np.sum(bitvector) / self.num_trans

			if support >= self.min_support:
				self.support_list[frozenset(sorted(prefix + [itemset]))] = np.sum(bitvector)

				suffix = []
				for itemset_sub, bitvector_sub in supportK:
					if self.use_CUDA:
						# bitvector = gpuarray.to_gpu(bitvector.astype(np.float32))
						# bitvector_sub = gpuarray.to_gpu(bitvector_sub.astype(np.float32))
						# union_bitvector = gpuarray.to_gpu((bitvector * bitvector_sub).get())
						# union_support = int(gpuarray.sum(union_bitvector).get()) / self.num_trans

						bitvector = bitvector.astype(np.float32)
						bitvector_sub = bitvector_sub.astype(np.float32)
						union_bitvector = np.zeros_like(bitvector)
						mod = SourceModule("""__global__ void multiply_element(float *dest, float *a, float *b, int len) {
											const int idx = threadIdx.x + blockDim.x * blockIdx.x;
											dest[idx] = a[idx] * b[idx];
										   }""")
						multiply = mod.get_function("multiply_element")

						multiply(cuda.Out(union_bitvector), cuda.In(bitvector), cuda.In(bitvector_sub), cuda.In(np.int32(self.num_trans)),
								 block=self.block,
								 grid=self.grid)
						union_support = gpuarray.sum(gpuarray.to_gpu(union_bitvector)).get() / self.num_trans
					else:
						union_bitvector = np.multiply(bitvector, bitvector_sub)
						union_support = np.sum(union_bitvector) / self.num_trans
					if union_support >= self.min_support:
						suffix.append([itemset_sub, union_bitvector])

				self.run(prefix + [itemset], sorted(suffix, key=lambda item: np.sum(item[1]), reverse=True))

	def get_result(self):
		print()
		return self.support_list


###################
# OUTPUT HANDLING #
###################
"""
	- Format the result from the recursive method for the output writer.
	- Input:
		- support_list: an unsorted dictionary recording the supports found by the recursive method.
	- Output:
		- L: the frequent itemset L, a 3-dimensional list
		- support_list: a dictionary recording the supports for each itemset in L
"""
def output_handling(support_list):
	L = []
	for itemset, count in sorted(support_list.items(), key=lambda x: len(x[0])):
		itemset = tuple(sorted(list(itemset), key=lambda x: int(x)))
		if len(L) == 0:
			L.append([itemset])
		elif len(L[-1][0]) == len(itemset):
			L[-1].append(itemset)
		elif len(L[-1][0]) != len(itemset):
			L[-1] = sorted(L[-1])
			L.append([itemset])
		else: raise ValueError()
	if len(L) != 0: L[-1] = sorted(L[-1])
	L = tuple(L)
	support_list = dict((tuple(sorted(list(k), key=lambda x: int(x))), v) for k, v in support_list.items())
	return L, support_list


#########
# ECLAT #
#########
"""
	- Implementation of the Eclat algorithm: large itemset counting and generation
	  with both the itreative method and recursive method implemented.
	- Input: 
		- data: M x N matrix with M = number of transactions, N = number of items in each transaction
		- min_support: minimum support
	- Output:
		- L: the frequent itemset L, a 3-dimensional list
		- support_list: a dictionary recording the supports for each itemset in L
"""
def eclat(data, min_support, iterative=False, use_CUDA=False):

	num_trans = float(len(data))
	vb_data, idx2item = compute_vertical_bitvector_data(data)
	
	if iterative and not use_CUDA:
		#---iterative method---#
		L1, support_list = compute_L1(vb_data, idx2item, num_trans, min_support)
		L = [L1]
		k = 1
		
		while True:
			print('Running Eclat: the %i-th iteration with %i itemsets in L%i...' % (k, len(L[-1]), k))
			k += 1
			LK, supportK = compute_LK(L[-1], support_list, k, num_trans, min_support)

			if len(LK) == 0:
				L = [sorted([tuple(sorted(itemset)) for itemset in LK]) for LK in L]
				support_list = dict((tuple(sorted(k)), np.sum(v)) for k, v in support_list.items())
				print('Running Eclat: the %i-th iteration. Terminating ...' % (k-1))
				break
			else:
				L.append(LK)
				support_list.update(supportK)
		return L, support_list

	elif not iterative:
		#---convert vertical bit vector matrix to dict then to list---#
		supportK = {}
		for idx, bit_list in enumerate(vb_data):
			supportK[idx2item[idx]] = bit_list
		supportK = sorted(supportK.items(), key=lambda item: np.sum(item[1]), reverse=True)
		
		#---recursive method---#
		if use_CUDA and not CUDA_FLAG: use_CUDA = False
		eclat = eclat_runner(num_trans, min_support, use_CUDA)
		eclat.run([], supportK)

		support_list = eclat.get_result()
		L, support_list = output_handling(support_list)

		return L, support_list
	else:
		raise NotImplementedError("Iterative with GPU is not yet implemented!")

