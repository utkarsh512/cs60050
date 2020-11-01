## Naive Bayes Classifier
### Important Libraries
Following libraries must be installed on the system to run `main.py`:
* `numpy` for mathematical operations
* `matplotlib` for plotting graphs
* `pandas` for handling dataset
* `sklearn` for encoding categorical variables and PCA

### I/O Handling
`main.py` doesn't print anything on the terminal (except the output from the Backward selection method), instead `FileIO` is used.
Following files are used for `IO` handling:
* `output.txt` used in `w` mode for printing the program output as specified in the problem statement.
* `output_detailed.txt` used in `w` mode for printing the details of each step during the execution of program.

All these files are generated in the same folder where this code resides.
`Train_B.csv` must be present within the same folder as well.

### Executing the code
* Simply run the `main.py`.
* From this point onwards, all the tasks are performed which are given in the problem statement.
* It will take about 15 minutes for the program to terminate. So, be patient.
* As the program ends, plots will be generated in the same folder.
  * `plot_1c_cross_val.png`: Plot of accuracy observed during cross-validation (Part 1c)
  * `plot_1c_final_acc.png`: Scatter plot of true and predicted value on test set (Part 1c)
  * `plot_2a_scree_graph.png`: Scree graph for principal component in PCA (Part 2a)
  * `plot_2a_variance_proportion_explained.png`: Line plot of variance preserved after selecting first `k` principal components. (Part 2a)
  * `plot_2b_cross_val.png`: Plot of accuracy observed during cross-validation after tranforming the samples as per PCA (Part 2b)
  * `plot_3b_accuracy_backselection.png`: Plot of increase in accuracy as the features are dropped from backward feature selection algorithm. (Part 3b)
  * `plot_3d_cross_val.png`: Plot of accuracy observed during cross-validation on best features. (Part 3d)
  * `plot_3d_final_acc.png`: Scatter plot of true and predicted value on test set after selecting best features. (Part 3d)
