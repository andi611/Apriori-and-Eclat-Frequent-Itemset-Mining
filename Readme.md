# Data Mining: Apriori and Eclat Frequent Itemset Mining
**Implementation of the Apriori and Eclat algorithms, two of the best-known basic algorithms for mining frequent item sets in a set of transactions, implementation in Python.**


## Requirements
* < Python 3.6+ >
* **< NVIDIA CUDA 9.0 >** (Optional)
* **< Pycuda 2018.1.1 >** (Optional)
* **< g++ [gcc version 6.4.0 (GCC)] >** (Optional)


## Environment Setup
* Install CUDA: [CUDA 9.0 installation guide](https://medium.com/@taylordenouden/installing-tensorflow-gpu-on-ubuntu-18-04-89a142325138)
* Install Pycuda:
```
sudo pip3 install pycuda
```
* Refer [here](https://github.com/ethereum-mining/ethminer/issues/731) for "CUDA unsupported GNU version" problem, or follow the following steps:
```
1. sudo apt-get install gcc-6
2. sudo apt-get install g++-6
3. sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-6 10
4. sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-6 10
```


## Implementaions
* Apriori algorithm
* Eclat algorithm (recursive method w/ **GPU acceleration** support)
* Eclat algorithm (iterative method)


## Usage
* To run the Apriori / Cclat algorithm with defaul settings:
```
python3 runner.py apriori
python3 runner.py eclat
```

* To run Eclat with GPU acceleration:
```
python3 runner.py eclat --use_CUDA --min_support 0.05 --input_path ./data/data2.txt
```
where 'data2.txt' is a harder dataset that requires more computational power.

* Other arguments can be given by:
```
python3 runner.py [mode] --min_support 0.6 --input_path ./data/data.txt --output_path ./data/output.txt
```

* To plot run time v.s. different minimum support value:
```
python runner.py [mode] --plot
```

* To test with toy data:
```
python runner.py [mode] --toy_data
```

* To run the eclat algorithm with the iterative method:
```
python runner.py [mode] --iterative
```


## Apriori minimum support v.s. run time plot
![](https://github.com/andi611/dataMining_apriori_eclat_freqItemsetMining/blob/master/data/plot_apriori.jpeg)


## Eclat minimum support v.s. run time plot
![](https://github.com/andi611/dataMining_apriori_eclat_freqItemsetMining/blob/master/data/plot_eclat.jpeg)


## Reference
* [PyCUDA tutorial documentation](https://documen.tician.de/pycuda/tutorial.html)
* [PyCUDA array documentation](https://documen.tician.de/pycuda/array.html)
* [PyCUDA tutorial](https://blog.csdn.net/u012915829/article/details/72831801)
* [CUDA parallel thread hierarchy](https://devblogs.nvidia.com/even-easier-introduction-cuda/cuda_indexing/)
* ![](https://github.com/andi611/DataMining_Apriori_Eclat_FreqItemsetMining/blob/master/data/cuda_indexing.png)
	 CUDA executes kernels using a grid of blocksof threads. This figure shows the common indexing pattern used in CUDA programs using the CUDA keywords gridDim.x (the number of thread blocks), blockDim.x (the number of threads in each block), blockIdx.x (the index the current block within the grid), and threadIdx.x (the index of the current thread within the block).