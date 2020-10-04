## Regression Decision Tree Classifier
### Important Libraries
Following libraries must be installed on the system to run `main.py`:
* `numpy` for mathematical operations
* `matplotlib` for plotting graphs
* `pandas` for handling dataset

### I/O Handling
`main.py` doesn't print anything on the terminal, instead `FileIO` is used.
Following files are used for `IO` handling:
* `PercentageIncreaseCOVIDWorldwide.csv` used in `r` mode for extracting the dataset
* `data.txt` used in `w` mode for printing the program output. All the output is written
in this file only.
* `tree.txt` used in `w` mode for printing the best depth tree (without pruning).
* `prunedTree.txt` used in `w` mode for printing the best depth tree (with pruning).

All these files except `PercentageIncreaseCOVIDWorldwide.csv` are generated in the same folder where this code resides.
`PercentageIncreaseCOVIDWorldwide.csv` must be present within the same folder as well.

### Executing the code
* Simply run the `main.py`.
* The program will ask to the user to input `max_depth`. Enter the depth.
* From this point onwards, all the tasks are performed which are given in the problem statement.
* It will take about 10 minutes for the program to terminate. So, be patient.
* As the program ends, plots will be generated in the same folder.
    * `tree.png` will contain the plot of `mean squared error` observed for passed `max_depth` (`Question 1`).
    * `analyze.png` will contain the plot of `mean squared error` observed for different values of `max_depth` (`Question 2`).
