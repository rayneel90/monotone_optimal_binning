"""
Author: Nilabja Ray
Date: 17May2022

This module implements coarse binning of all continuous features. The algorithm
satisfies three major requirements of good bins, viz.
1. The bins are Monotonic in terms of bad rates and WOE
2. The bins provide maximum possible correlation with the target, subject to a
    manually set threshold ( Setting a low threshold may result in too few bins)
3. Each bin satisfies user-set constraints of minimum number of observations and
    minimum bad rate

Current version has following draw-backs
1. Categorical variables are ignored and not binned. It is possible to get
    WOE-binning for categorical variables, although the concept of monotonicity
    does not exist. In future version we will implement binning of categorical
    variables which follow point 2 and 3 mentioned above
2. Does not handle unclean target variable. For current version target must be
    encoded as 0 and 1 where 1 is considered as the target-class. Also presence
    of missing value in target raises error. In future we will implement missing
    value handling and target encoding in the module.
3. Though numpy operations are by default multi-threaded, there is still scope
    of optimising the process further using multiprocessing. Depending on the
    speed achieved on different testing dataset, multiprocessing may be
    implemented in future versions.
"""
import os
import pandas as pd
from scipy.stats import pointbiserialr
import warnings
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from statsmodels.stats.proportion import proportions_ztest as prop_test


class WoeIv(BaseEstimator, TransformerMixin):
    """
    The class extends from sklearn baseEstimator to provide sklearn-like fit,
     transform and fit-transform API. As a result, this can be used as part of
     an sklearn pipeline.
    """

    def __init__(self, min_sample_fraction: float = 0.05,
                 min_target_rate: float = 0.01, p_threshold: float = 0.25,
                 verbose: bool = False):
        """
        Initialises the class with user-defined thresholds for the WOE algorithm
        :param min_sample_fraction: Each bin will contain at least this fraction
            of train dataset. Default value 5%.
        :param min_target_rate: Min target rate that will be maintained in each
            bin. Default value 1%.
        :param p_threshold: Threshold p-value to be used as a stopping rule
            while optimising the bins. setting this too low will result in very
            few bins. default value 0.25
        :param verbose: whether to print steps in console. Setting True is
            recommended for large data-sets which takes a lot of time to process
        """
        self.smpl_thresh: float = min_sample_fraction
        self.trgt_thresh: float = min_target_rate
        self.p_threshold: float = p_threshold
        self.verbose: bool = verbose

        self.y: pd.Series = object
        self.X: pd.DataFrame = object
        self.monotonic_bins: dict = {}
        self.optimized_bins: dict = {}
        self.woe_details: dict = {}
        self.iv_dict: dict = {}

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        :param X: pandas DataFrame containing features. If X contains
            non-numeric features, they will be dropped. Any numeric feature with
            less than 6 distinct values is considered as categorical and hence
            dropped.
        :param y: pandas Series of target. Target must be dichotomous and
            should be label-encoded as 0 and 1. 1 is considered as target.
            Presence of missing value in y will raise error.
        :return: None
        """
        # raise error if y is not pandas series
        assert isinstance(y, pd.Series), \
            "y must be pandas.Series, {} provided instead".format(type(y))

        # raise error if y is not dichotomous
        assert y.nunique() == 2, "y must be dichotomous"

        # raise error is y contains missing value
        assert y.isna().sum() == 0, \
            "target contains missing value. Remove all observations with " \
            "missing target from your dataset and then fit WOE"
        # raise error if X is not pandas DataFrame
        assert isinstance(X, pd.DataFrame), \
            "X must be pandas.Dataframe, {} provided instead".format(type(X))

        if self.verbose == 1:
            "Running WOE Binning on all numeric variables"
        self.X = X.select_dtypes("number")  # drop non-numeric features
        self.y = y.rename('target')

        # Since WOE calculations of different columns is independent, it can be
        # done in loop. Future versions will implement multiprocessing for
        # this step
        for colnm in self.X.columns:
            if self.verbose:
                print("creating monotonic bins for {}".format(colnm))
            tmp_df, null_df, asc, bins = self.create_monotonic_bin(colnm)
            if tmp_df is None:
                continue
            if self.verbose:
                print("Optimizing bins for {}".format(colnm))
            self.__optimize_p(colnm, tmp_df, asc, bins)

            # Create separate bin for missing values
            if not null_df.empty:
                self.optimized_bins[colnm] = self.optimized_bins[colnm].append(
                    pd.Series(dict(
                        bin_thresh=np.nan,
                        bad_cnt=null_df.target.sum(),
                        nobs=null_df.shape[0])
                    ), ignore_index=True
                )
        # Compute WOE and IV for finalised bins
        self.__compute_woe_iv()

    def create_monotonic_bin(self, colnm) -> object:
        """
        Creates a binning which is monotonic in terms of bad rate.
        :param colnm: name of the feature which is binned
        :return: tmp_df: DataFrame created by stacking feature and target.
        Observations with missing feature are removed and returned separately.
        :return: null_df: Observations from tmp_df with missing feature
        :return: asc: Indicator of whether feature has increasing or decreasing
            relation with target. True indicates decreasing relation.
        :return: bins: Threshold of monotonic bins
        """
        tmp_df = pd.concat([self.X[colnm].astype(float), self.y], axis=1)
        null_df = tmp_df[tmp_df.isna().any(axis=1)].copy()
        tmp_df.dropna(how='any', inplace=True)
        tmp_df.sort_values(colnm, inplace=True)

        # If missing value more than (1-min_sample_fraction), skipping binning
        if null_df.shape[0] / self.X.shape[0] >= (1 - self.smpl_thresh):
            if self.verbose:
                print("Feature {} contains more than {:.0%} missing values. "
                      "Skipping binning.".format(colnm, 1 - 2 * self.smpl_thresh))
            return None, None, None, None
        if tmp_df[colnm].value_counts(normalize=True).max() >= \
                (1 - self.smpl_thresh):
            if self.verbose:
                print(
                    "more than {:.0%} observations concentrated against single "
                    "value. Skipping binning".format(1 - 2 * self.smpl_thresh)
                )
            return None, None, None, None

        # asc if the indicator whether the variable has increasing or decreasing
        # relation with target. As a proxy of monotonicity, which is rare with
        # practical dataset, we check the bad ratio against lowest 25% and
        # highest 25% of the feature value. asc is True if the relation is
        # decreasing, i.e. bad rate is higher for lower value of feature.
        asc = tmp_df.iloc[:tmp_df.shape[0] // 4].target.mean() > \
              tmp_df.iloc[3 * tmp_df.shape[0] // 4:].target.mean()

        # begin with each observation as separate bin
        bins = np.sort(tmp_df[colnm].unique())
        i = 0
        while 1:
            if self.verbose and i == 0:
                print("Iter 0: feature {} contains {} distinct values".format(
                    colnm, len(bins)
                ))
            if asc:
                # for asc=True, "bins" indicate lower thresholds of each bins
                # and contains -inf
                if -np.inf not in bins:
                    bins = np.insert(bins, 0, -np.Inf)
                cut_points = np.append(bins, np.Inf)
            else:
                # for asc=False, "bins" indicate upper thresholds of each bins
                # and contains inf
                if np.inf not in bins:
                    bins = np.append(bins, np.Inf)
                cut_points = np.insert(bins, 0, -np.Inf)

            # bin the feature and compute bad rate and size for each bin
            tmp_df['bin_thresh'] = pd.cut(
                tmp_df[colnm], bins=cut_points, labels=bins).astype(float)
            init_summary = tmp_df.groupby('bin_thresh')['target'].agg(["mean", "size"])

            # convert bin thresholds from index to a column
            init_summary.reset_index(inplace=True)

            # order dataframe depending on whether the relation between feature
            # and target is increasing or decreasing. For decreasing relation,
            # lowest bin of feature comes at top and vice versa.
            init_summary.sort_values('bin_thresh', ascending=asc, inplace=True)

            # Monotonicity is maintained if bad rate in each bin is higher than
            # the subsequent bin. Identify all bins where this is getting
            # violated
            non_mono = init_summary[
                init_summary['mean'] > init_summary['mean'].shift(1).fillna(1)].copy()

            if non_mono.empty:
                if self.verbose:
                    print({"Monotonicy Achived. {} bins created.".format(
                        init_summary.shape[0])})
                break  # end iteration if all bins maintain monotonicity
            i = i + 1
            if self.verbose:
                print("Iter {}: removing {} non monotonic bins".format(
                    i, non_mono.shape[0]))
            # Remove thresholds corresponding to all non-monotonic bins
            bins = np.sort(list(set(bins).difference(non_mono.bin_thresh)))

        # final threshold for all monotonic bins
        bins = np.sort(init_summary.bin_thresh)

        # store monotonic bins in class attribute
        self.monotonic_bins[colnm] = init_summary
        return tmp_df, null_df, asc, bins

    def __optimize_p(self, colnm, tmp_df, asc, bins):
        """
        Optimize/reduce bins through pair-wise test of similarity of bad rate
        between consecutive bins. Large sample normality approximation by
        CLT is used for the test of hypothesis.
        :param colnm: name of feature
        :param tmp_df: raw dataframe obtained by stacking feature and target and
            removing null values in feature
        :param asc: indicator of direction of monotonicity. Check docstring of
            __create_monotonic_bin method for details.
        :param bins: thresholds of monotonic bins
        :return: None
        """
        i = 0
        while 1:

            if asc:
                # for asc=True, "bins" indicate lower thresholds of each bins
                # and contains -inf
                if -np.inf not in bins:
                    bins = np.insert(bins, 0, -np.Inf)
                cut_points = np.append(bins, np.Inf)
            else:
                # for asc=False, "bins" indicate upper thresholds of each bins
                # and contains inf
                if np.inf not in bins:
                    bins = np.append(bins, np.Inf)
                cut_points = np.insert(bins, 0, -np.Inf)

            # Create bins and compute bad count and n_observations
            tmp_df['bin_thresh'] = pd.cut(
                tmp_df[colnm], bins=cut_points, labels=bins
            ).astype(float)
            init_summary = tmp_df.groupby('bin_thresh')['target'].agg(
                ["sum", "size"]
            )
            init_summary.columns = ['bad_cnt', 'nobs']

            # move bin thresholds from from index to column
            init_summary.reset_index(inplace=True)

            # order dataframe depending on whether the relation between feature
            # and target is increasing or decreasing. For decreasing relation,
            # lowest bin of feature comes at top and vice versa.
            init_summary.sort_values('bin_thresh', ascending=asc, inplace=True)

            # storing a copy of the binned data in class attribute
            self.optimized_bins[colnm] = init_summary.copy()

            # add the previous bin's bad count and n_observations as separate col
            init_summary['prev_bad_cnt'] = init_summary['bad_cnt'].shift()
            init_summary['prev_nobs'] = init_summary['nobs'].shift()

            # drop the first bin since there is nothing to compare it with
            init_summary.dropna(inplace=True)

            # p value for pairwise test of equality of bade rate. Since the bins
            # are already monotonic, the test shall be one-sided.
            init_summary['p_val'] = init_summary.apply(
                lambda x: prop_test(
                    x.filter(like="bad_cnt"), x.filter(like="nobs"),
                    alternative='smaller'
                )[1], axis=1).fillna(1)
            # missing p-value is due to too few no of obs in bin. Since such
            # bins should always be clubbed, we replace them with 1, which
            # ensures they are beyond the p-value threshold.

            # for classes where either of the user defined thresholds of min obs
            # fraction or min bad fraction is not met, manually increase p-val
            # by 1. Increasing p-val by such a large value will ensure that it
            # is above p-val threshold and hence will always get combined.
            init_summary['p_val'] = \
                init_summary['p_val'] + (
                        (init_summary['bad_cnt'] / self.y.sum() < self.trgt_thresh) |
                        (init_summary['prev_bad_cnt'] / self.y.sum() < self.trgt_thresh) |
                        (init_summary['nobs'] / self.y.shape[0] < self.smpl_thresh) |
                        (init_summary['prev_nobs'] / self.y.shape[0] < self.smpl_thresh)
                ).astype(int)
            # compute the max of p values and remove the corresponding bin if
            # the value is more than the user-defined threshold. If max p-value
            # is less than the threshold, it means that the optimality has been
            # achieved and the iteration shall stop
            max_p = init_summary['p_val'].max()
            i = i + 1
            if self.verbose:
                print("Iteration {}: max p {:.2f}".format(i, max_p))
            if max_p > self.p_threshold:
                bins = np.sort(
                    init_summary[init_summary.p_val != max_p].bin_thresh.values)
            else:
                break
            # If only two bins remain, exit loop
            if len(bins) == 0:
                break
        if asc:
            bins = np.sort(self.optimized_bins[colnm].bin_thresh)
            bins = np.insert(np.delete(bins, 0), 0, -np.inf)
            self.optimized_bins[colnm].bin_thresh = pd.IntervalIndex.from_arrays(
                bins, np.append(np.delete(bins, 0), np.Inf))
        else:
            bins = np.sort(self.optimized_bins[colnm].bin_thresh)
            bins = np.append(np.delete(bins, -1), np.inf)
            self.optimized_bins[colnm].bin_thresh = pd.IntervalIndex.from_arrays(
                np.insert(np.delete(bins, -1), 0, -np.Inf), bins)
        if self.verbose:
            print("Optimisation complete for feature {}. Created {} bins".format(
                colnm, len(bins)
            ))
        return None

    def __compute_woe_iv(self):
        """
        Compute WOE and IV against each bin of each variable. Current version
        iterates over the features in a single-threaded loop, which can be
        parallelized for further speed optimization
        :return: None
        """
        for key, val in self.optimized_bins.items():
            tmp = val.copy()
            tmp['good_cnt'] = tmp.nobs - tmp.bad_cnt
            tmp['bad_rate'] = (tmp.bad_cnt / tmp.bad_cnt.sum())
            tmp['good_rate'] = (tmp.good_cnt / tmp.good_cnt.sum())
            tmp['woe'] = np.log(tmp.bad_rate / tmp.good_rate)
            tmp['IV'] = (tmp.bad_rate - tmp.good_rate) * tmp.woe
            self.woe_details[key] = tmp
            self.iv_dict[key] = tmp.IV.sum()

    def get_woe_summary(self):
        """
        Returns WOE and IV details of each feature and each bin in a neat single
        table format
        :return: Dataframe containing bad rate, WOE and IV for all bins along
            with feature names
        """
        return pd.concat(self.woe_details).droplevel(1).reset_index().rename(
            columns={"index": "Feature"})

    def get_iv_summary(self):
        """
        Returns the summed up IV value of each feature in a single pandas series
        :return: Pandas Series of IV values with feature names as index
        """
        return self.get_woe_summary().groupby('Feature').IV.sum()

    def transform(self, X):
        """
        Bins the numeric columns and replace them with corresponding WOE values.
        Non-numeric columns as well as columns different from train feature-set
        are removed with an warning
        :param X: pandas DataFrame of columns
        :return: Pandas Dataframe of WOE-transformed columns
        """
        # Assert that X is a pandas DataFrame
        assert isinstance(X, pd.DataFrame), \
            "X must be pandas.Dataframe, {} provided instead".format(type(X))

        # Raise warning if X contains any column different from train features.
        # Since non-numeric columns are ignored during training, by default
        # warning is raised for those.
        if len(X.columns.difference(self.woe_details.keys())):
            warnings.warn(
                "WOE for following features in X were not computed during "
                "fitting. They will be skipped:\n{}".format(
                    ",".join(X.columns.difference(self.woe_details.keys()))
                )
            )

        col_lst = X.columns.intersection(self.woe_details.keys())

        # return pandas Dataframe of transformed Varu=iables
        return X[col_lst].apply(lambda x: pd.cut(
            x,
            pd.IntervalIndex(self.woe_details[x.name].bin_thresh.dropna())
        ).astype("interval").map(dict(zip(
            self.woe_details[x.name].bin_thresh,
            self.woe_details[x.name].woe
        ))))
