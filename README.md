In this assignment you will implement the K-Nearest-Neighbor algorithm and a cross validation error estimate. The assignment has 2 phases:

Finding the best hyper parameters (K – number of neighbors, l-p distance measure, majority method) using 10-folds cross validation, for 2 different datasets ("glass" & "cancer").
In this phase you need to go over all hyper parameters combination, and for each combination to calculate the average error over all 10 folds.

The possible values for the hyper parameters are:

K – {1, 2, …, 20}
l-p distance – {infinity, 1, 2, 3}
Majority – {"uniform", "weighted"}
For each dataset you will chose the combination with the lower average error and print this combination and the average cross validation error.

In addition, for the cancer dataset only you will print the average Precision & Recall among the 10 folds. This dataset has 2 classes – recurrence-events, no-recurrence-events which converted to 0.0, 1.0 appropriately by WEKA. For this homework the 0 class (recurrence-events) will be the positive class and the 1 class (no-recurrence-events) will be the negative class.
