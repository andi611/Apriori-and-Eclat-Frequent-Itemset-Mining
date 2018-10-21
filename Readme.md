# Data Mining: Apriori and Eclat Frequent Itemset Mining
**Implementation of the Apriori and Eclat algorithms, two of the best-known basic algorithms for mining frequent item sets in a set of transactions, implementation in Python.**

## Implementaions
* Apriori algorithm
* Eclat algorithm (recursive method)
* Eclat algorithm (iterative method)

## Usage
* To run the apriori / eclat algorithm with defaul settings:
```
python3 runner.py apriori
python3 runner.py eclat
```

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