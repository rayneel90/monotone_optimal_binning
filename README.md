# monotone_optimal_binning 
An optimized coarse binning algorithm for calculation of WOE and IV
This module implements the algorithm proposed by [Pavel Mironchyk](https://www.researchgate.net/profile/Pavel-Mironchyk-3) and [Viktor Tchistiakov](https://www.researchgate.net/profile/Viktor-Tchistiakov) in their paper [Monotone optimal binning algorithm for credit risk modeling](https://www.researchgate.net/publication/322520135_Monotone_optimal_binning_algorithm_for_credit_risk_modeling/link/5a5dd1a8458515c03edf9a97/download). 

There are other implementations of the same algorithm which are found [here](https://github.com/jstephenj14/Monotonic-WOE-Binning-Algorithm) and [here](https://cemsarier.github.io/algorithm/credit%20scoring/scorecard/woe_binning/). Current implementation is an improvement over these in two major aspects:
1. It uses Native Pandas and Numpy functions to boost performance, compared to the multiple loops being used in other implementations.
2. It works on a complete training data-set instead of computing WOE for a single feature

Additionally, it provides a sklearn-like API where users can call `.fit()` method to train the WOE bins and then use `.transform()` method to apply those WOE bins to new datasets. 


