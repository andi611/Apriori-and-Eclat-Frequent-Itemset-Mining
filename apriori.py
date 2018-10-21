# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ apriori.py ]
#   Synopsis     [ Implement of the Apriori algorithm]
#   Author       [ Ting-Wei Liu (Andi611) ]
#   Copyright    [ Copyleft(c), NTUEE, NTU, Taiwan ]
"""*********************************************************************************************"""


#############################
# COMPUTE C1 AND L1 ITEMSET #
#############################
"""
	- Derivation of large 1-itemsets L1: 
		- At the first iteration, scan all the transactions and count the number of occurrences for each item.
	- Input: 
		- data: M x N matrix with M = number of transactions, N = number of items in each transaction
		- min_support: minimum support
	- Output:
		- output1: a list of frozensets containing the candidates in C1
		- output2: a dictionary recording the supports for each candidates in C1
"""
def compute_C1_and_L1_itemset(data, num_trans, min_support):
	#---compute C1---#
	C1 = {}
	for transaction in data:
		for item in transaction:
			if not item in C1:
				C1[item] = 1
			else: C1[item] += 1
	#---compute L1---#
	L1 = []
	support1 = {}
	for candidate, count in sorted(C1.items(), key=lambda x: x[0]):
		support = count / num_trans
		if support >= min_support:
			L1.insert(0, [candidate])
			support1[frozenset([candidate])] = count
	return list(map(frozenset, sorted(L1))), support1, C1


##############
# COMPUTE CK #
##############
"""
	- Performs data base scan and compute the next candidate set CK from LK-1,
	  At the k-th iteration, the candidate set Ck are those whose every (k-1) item subset is in Lk-1. 
	  (k-1) is inclusive, so in implementation we need to set index to [:k-2]
	- Input:
		- LK_: the large K-1 itemset, LK-1
		- k: the k-th iteration
	- Output:
		- CK: the candidate set CK for the k-th iteration
"""
def compute_CK(LK_, k):
	CK = []
	for i in range(len(LK_)):
		for j in range(i+1, len(LK_)): # enumerate all combinations in the Lk-1 itemsets
			L1 = sorted(list(LK_[i]))[:k-2]
			L2 = sorted(list(LK_[j]))[:k-2]
			if L1 == L2: # if the first k-1 terms are the same in two itemsets, merge the two itemsets
				new_candidate = frozenset(sorted(list(LK_[i] | LK_[j]))) # set union
				CK.append(new_candidate) 
	return sorted(CK)


##############
# COMPUTE LK #
##############
"""
	- Scan data set and compute the support by counting the # of occurrences for each candidate itemset.
	  Lk is the group of candidates in Ck that is a subset to one of the items in data set D.
	- Input:
		- D: the set containing distinct items in the data set
		- CK: the candidate set CK for the k-th iteration
		- min_support: minimun support
	- Output:
		- LK: the large K itemset
		- supportK: a dictionary recording the supports for each itemsets in LK
"""
def compute_LK(D, CK, num_trans, min_support):
	support_count = {}
	for item in D: # traverse through the data set
		for candidate in CK: # traverse through the candidate list
			if candidate.issubset(item): # check if each of the candidate is a subset of each item
				if not candidate in support_count:
					support_count[candidate] = 1
				else: support_count[candidate] += 1
	LK = []
	supportK = {}
	for candidate, count in sorted(support_count.items(), key=lambda x: x[0]):
		support = count / num_trans
		if support >= min_support:
			LK.append(candidate)
			supportK[candidate] = count
	return sorted(LK), supportK


###########
# APRIORI #
###########
"""
	- Implementation of the Apriori algorithm: large itemset counting and generation
	- Input: 
		- data: M x N matrix with M = number of transactions, N = number of items in each transaction
		- min_support: minimum support
	- Output:
		- LK: the frequent itemset LK
		- supportK: a dictionary recording the supports for each candidates in C1
"""
def apriori(data, min_support):
	D = sorted(list(map(set, data)))
	num_trans = float(len(D))
	L1, support_list, CK = compute_C1_and_L1_itemset(data, num_trans, min_support)
	L = [L1]
	k = 1

	while (True): # create superset k until the k-th set is empty (ie, len == 0)
		print('Running Apriori: the %i-th iteration with %i candidates...' % (k, len(CK)))
		k += 1
		CK = compute_CK(LK_=L[-1], k=k)
		LK, supportK = compute_LK(D, CK, num_trans, min_support)
		if len(LK) == 0: 
			L = [sorted([tuple(sorted(list(itemset), key=lambda x: int(x))) for itemset in LK]) for LK in L]
			support_list = dict((tuple(sorted(list(k), key=lambda x: int(x))), v) for k, v in support_list.items())
			print('Running Apriori: the %i-th iteration. Terminating ...' % (k-1))
			break
		else:
			L.append(LK)
			support_list.update(supportK)
	return L, support_list

