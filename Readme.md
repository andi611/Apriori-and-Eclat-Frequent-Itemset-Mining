# Data Mining: Apriori and Eclat Frequent Itemset Mining
**Implementation of the Apriori and Eclat algorithms, two of the best-known basic algorithms for mining frequent item sets in a set of transactions, implementation in Python.**


## Implementaions
* Apriori algorithm
* Eclat algorithm (recursive method w/ **GPU acceleration** support)
* Eclat algorithm (iterative method)


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

## Datasets:
* **./data/data.txt**: suggested min support range: [0.6 0.02]
* **./data/data2.txt**: a harder dataset, only eclat can find results in reasonable time. Suggested min support range: [0.1 0.0002]


## Usage
* To run the Apriori / Cclat algorithm with defaul settings:
```
python3 runner.py apriori
python3 runner.py eclat
```

* Other arguments can be given by:
```
python3 runner.py [mode] --min_support 0.6 --input_path ./data/data.txt --output_path ./data/output.txt
```

* To run Eclat with **GPU acceleration** (Suggested dataset: data2.txt):
```
python3 runner.py eclat --min_support 0.02 --input_path ./data/data2.txt --use_CUDA
```

* To plot run time v.s. different experiment values:
```
python runner.py [mode] --plot_support
python runner.py [mode] --plot_support_gpu --input_path ./data/data2.txt --use_CUDA
python runner.py [mode] --compare_gpu --input_path ./data/data2.txt --use_CUDA
python runner.py [mode] --plot_thread --input_path ./data/data2.txt --use_CUDA
python runner.py [mode] --plot_block --input_path ./data/data2.txt --use_CUDA
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

## Eclat minimum support v.s. run time plot **(data2.txt w/ GPU version)**
![](https://github.com/andi611/DataMining_Apriori_Eclat_FreqItemsetMining/blob/master/data/plot_eclat_support_vs_execution_time.jpeg)
![](https://github.com/andi611/DataMining_Apriori_Eclat_FreqItemsetMining/blob/master/data/plot_eclat_support_vs_execution_time2.jpeg)

## Eclat w/ GPU and w/o GPU comparison plot **(data2.txt w/ GPU version)**
![](https://github.com/andi611/DataMining_Apriori_Eclat_FreqItemsetMining/blob/master/data/plot_compare_gpu.jpeg)

## Reference
* [PyCUDA tutorial documentation](https://documen.tician.de/pycuda/tutorial.html)
* [PyCUDA array documentation](https://documen.tician.de/pycuda/array.html)
* [PyCUDA tutorial](https://blog.csdn.net/u012915829/article/details/72831801)
* [CUDA parallel thread hierarchy](https://devblogs.nvidia.com/even-easier-introduction-cuda/cuda_indexing/)
* ![](https://github.com/andi611/DataMining_Apriori_Eclat_FreqItemsetMining/blob/master/data/cuda_indexing.png)
	 CUDA executes kernels using a grid of blocksof threads. This figure shows the common indexing pattern used in CUDA programs using the CUDA keywords gridDim.x (the number of thread blocks), blockDim.x (the number of threads in each block), blockIdx.x (the index the current block within the grid), and threadIdx.x (the index of the current thread within the block).