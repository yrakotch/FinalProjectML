# FinalProjectML
# FinalProjectML 

What's in this project? (Based on a ML course by Pr. Lior Rokach, BGU)

This project includes the execution and the statistical comparison of these four algorithms:

1. COBRA - which proposes a new method for combining several initial estimators of the regression function. Instead of building a linear or convex optimized combination over a collection of basic estimators, it uses them as a collective indicator of the proximity between the training data and a test observation. Based on this paper: Biau, et al. "COBRA: A combined regression strategy." Journal of Multivariate Analysis 146 (2016): 18-28.

2. EWA - which obtains sharp oracle inequalities for convex aggregates via exponential weights, under general assumptions on the distribution of errors and on the functions to aggregate. In other words, it obtains a weighted average of the predictions of all the algorithms that form the ensemble. Based on this paper: Dalalyan, Arnak S., and Alexandre B. Tsybakov. "Aggregation by exponential weighting and sharp oracle inequalities." International Conference on Computational Learning Theory. Springer, Berlin, Heidelberg, 2007.

3. Boruta - which is built around a base algorithm (e.g: the random forest algorithm) with additional implementation for finding all relevant features and removing non-relevant ones. Boruta core idea is that a feature that is not relevant is not more useful for classification than its version with a permuted order of values. Based on this paper: Miron B. Kursa, Witold R. Rudnicki: “Feature Selection with the Boruta Package". Feature Selection with the Boruta Package. September 2010, Volume 36, Issue 11.

4. Adaboost - The well-known adaptive boosting meta-algorithm (formulated by Yoav Freund and Robert Schapire).

The statistical data analysis includes performing the Friedman test on the MSE metric results of the four algorithms and additional meta-learning model implementation using XGBoost that performs the binary classification task of determining whether an algorithm will be the best performing one (rank 1) given a predefined dataset’s meta-features.


Installation:
1. pip install pycobra
2. pip install Boruta


Dependencies:
1. Python 3.4+
2. numpy, scipy, scikit-learn, matplotlib, pandas, seaborn.


Execution:

1. First run Regression.py, which will train and test the algorithms contained in the "models" list (line 49) on 100 regression datasets. The script will output a file "results.csv" inside the "results" folder.

2. Run StatisticalTests.py to obtain results of a Friedman and post-hoc tests on the results collected by the Regression.py.

3. Run MetaLearning.py to train and test a meta-learning model (XGBoost) on the meta-features and results collected by Regression.py, including statistical tests and extraction of feature importance.


Hyper-Parameters tuned:

AdaBoost:
1. n_estimators - sets the number of estimators in the chosen ensemble method.
2. ccp_alpha - regularization parameter for the base estimators.
  
Cobra:
1. epsilon - for determining the "distance" between the initial estimators and the new estimator.
2. machine_list - list of estimator types to be considered by the algorithm.

Ewa:
1. beta - the "temperature" parameter, which is used to build the estimator fn based on data. (for further explanation, look at EWA reference above).
2. machine_list - list of estimator types to be considered by the algorithm.
  
Boruta:
1. n_estimators - sets the number of estimators in the chosen ensemble method.
2. ccp_alpha - regularization parameter for the base estimators.



References:

1. Biau, Fischer, Guedj and Malley (2016), COBRA: A combined regression strategy. Journal of Multivariate Analysis.
2. Dalalyan, Arnak S., and Alexandre B. Tsybakov. "Aggregation by exponential weighting and sharp oracle inequalities." International Conference on Computational Learning Theory. Springer, Berlin, Heidelberg, 2007.
3. Miron B. Kursa, Witold R. Rudnicki: “Feature Selection with the Boruta Package". Feature Selection with the Boruta Package. September 2010, Volume 36, Issue 11.
4. Guedj and Srinivasa Desikan (2020), Kernel-based ensemble learning in Python. Information.
5. Guedj and Srinivasa Desikan (2018), Pycobra: A Python Toolbox for Ensemble Learning and Visualization. Journal of Machine Learning Research.
