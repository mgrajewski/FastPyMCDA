# -*- coding: utf-8 -*-

"""
Created on Tue Jan 22 10:07:03 2019

This class contains the implementation of some decision models based upon
outperforming. It is assumed that the decision maker considers a variety of
criteria each of which we assign a certain benefit or value to choose from the
K alternatives given.
We group the criteria into a set of m superordinate categories. The i-th
category in turn consists of n(i) subcategories. Within the i-th category,
the perceived importance of each of the criteria is modelled by certain
weighting factors v_{i,j} wich are normalized such that
    v_{i,1} + ... + v_{i, n(i)} = 1.
We reflect the overall importance of the i-the category by assigning
weighting factors w_i, i = 1, ..., m and obtain the overall importance of the
j-th criterion in the i-th class by w_i * v_{i,j}. As within a category, we
normalize enforcing
    w_1 + ... + w_n = 1.
We sometimes refer to the w_i as categorial weights and to the v_{i,j} as
subweights.

For the k-th alternative, the values of each criterion in each category are
described by u_{i,j}^k. The two decision models implemented (Promethee II and
extended SAW), differ in the way how to compute the values
u_{i,j}^k from the characteristics x_{i_j}^k which must be provided by the
user.
In extended SAW (simple additive weighting), we obtain the values u_{i,j}^k by
simply scaling the characteristics x_{i,j}^k such that
    \sum_k |u_{i,j}^k| = 1.
Promethee II, however is more sophisticated here. We refer to the literature
for further details.    

For each alternative, we calculate its performance index p_k computing
    p_k = \sum_{i=1}^m w_i*(\sum_{j = 1}^{n(i)} v_{i,j} u_{i,j}^k ).
We choose the alternative with the maximal performance index.

We refer to both the weights and the values as model parameters.

It is commonplace to require w_1 + ... + w_m = 1. We relax this to
require w_i >= 0, as scaling to achieve this side condition replaces
w = (w_1, ..., w_m) by c*w, the real number c chosen appropriately. However,
the same c applies to all alternatives such that all performance indices are
scaled by the same c which preserves their ranking and therefore the outcome of
the model. Thus, we can skip scaling at all, if only the alternatives chosen
are required. This, however, does not apply to the scaling within one category.

When applied to our studies of e-mobility, there are three alternatives: 
    0: full e-mobility
    1: internal combustion engines
    2: hybrid cars

@author: Simon Grasse, Forschungszentrum Jülich
Matthias Grajewski, FH Aachen University of Applied Sciences
Stefan Vögele, Forschungszentrum Jülich
"""

import sys
import os
import warnings
import pandas as pd
import numpy as np

# make sure root package-directory is in python search-path
if '__file__' in globals():
    ABS_DIR_NAME = os.path.abspath(os.path.dirname(__file__))
else:
    ABS_DIR_NAME = os.getcwd()

if os.path.abspath(os.path.join(ABS_DIR_NAME, '..', '..')) not in sys.path:
    sys.path.append(os.path.abspath(os.path.join(ABS_DIR_NAME, '..', '..')))

from pymcda.src import my_utils as u


class Data:
    """Contains all methods to read and evaluate desired model parameters.

    On creation it reads a set of model parameters from an Excel file and
    creates a `pandas` `DataFrame` to store the data read after being normalized.
    The parameter set can be changed by accessing the property associated with
    `_dframe_base`. When doing so, numerical values for weighting factors and
    characteristics have to be provided by the user.

    Attributes
    ----------
    _dframe_base : pandas DataFrame-object containing the parameter set.

    Methods
    -------
    get_class_and_scores()
        Evaluates extended SAW for two variable categorial weighting
        factors (their numerical values given as two lists) and returns the
        indices of the alternatives chosen as well as the performance indices
        of all alternatives.

    get_class_and_scores_P2()
        Evaluates Promethee II for two variable categorial weighting
        factors (their numerical values given as two lists) and returns the
        indices of the alternatives chosen as well as the performance indices
        of all alternatives.
        
    normalize_within_categories(self, weight):
        This function returns subweights normalized within a category aka
        it scales all weighting factors belonging to a category such that their
        sum is one.
        
    normalize_values(self, dframe):
        This function enforces \sum_k |u_{i,j}^k| = 1 for the given DataFrame
        dframe.
        
    print_dframe()
        Prints the DataFrame `_dframe_base`.
        
    get_dataframe()
        Read Parameter set from Excel-file(xlsx) and construct a pandas
        `DataFrame` _dframe_base
    """

    def __init__(self, file_name, sheet_name):
        """
        During initialization, we read the set of model parameters from a
        certain worksheet of an Excel file.

        Parameters
        ----------
        file_name : str
            Pathname to an Excel file (suffix .xlsx) containing the numerical
            values of the model parameters. The Excel file may consist of
            several worksheets.
        sheet_name : str
            Name of the Excel worksheet to read the numerical values of the
            model parameters from
        """

        self._dframe_base = self.get_dataframe(file_name, sheet_name)

# -----------------------------------------------------------------------------

    def get_class_and_scores(self, names_categories, weights_categories, return_alternatives):
        """
        In our investigations, we consider some categorial weights as
        variables. This function returns the result of the decision model aka
        the index of the most preferrable alternative for various weights given
        in weights_categories of selected categories provided in
        names_categories. The weights for all other categories are taken from
        the dframe_base-object.

        When called, `get_class_and_scores` will do a deepcopy of the
        `self.__dframe_base` containing the parameter set.
        The best performing alternative will be then calculated by comparing
        the sum of all criteria values multiplied by their corresponding total
        weight (for details, we refer to the explanation above).
        
        The existence of return_alternatives is for performance reasons: we do not
        need to normalize the categorial weights if we are interested in the
        indices of the alternatives only and can thus skip this computation for
        return_alternatives=False.

        Parameters
        ----------
        names_categories : List of strings
            Names of categories considered variable
        weights_categories : numpy array containing the numerical values of the
            corresponding weighting factors (these are NOT the values of the
            criteria grouped in the categories!)
        return_alternatives: boolean
            if true, return the performance indices for all alternatives and
            the indices of the best performing alternatives, if false, return
            the indices of the alternatives only

        Returns
        -------
        integer list
            Indices of the best performing alternatives
            
        double list
            performance indices for any alternative for the weighting factors
            given in weights_categories (only for return_alternatives = True)
        """

        # deep copy is a performance hit, but it keeps the changes local.
        # Directly working on self._dframe_base by just typing
        # dframe = self._dframe_base
        # implicitly changes values in the pandas object which can create
        # nasty side effects when subsequently changing different weights in a
        # more complex analysis
        dframe = self._dframe_base.copy()

        # vector containing the values with respect to all criteria for all
        # options
        option_values = dframe.loc[:,('option_values')]
        option_values = option_values.to_numpy()
          
        weights_within_categories = dframe.loc[:,('weights_values', 'subweight')].to_numpy()
        numberOfValues = weights_categories[:,0].size
        
        # preallocate result arrays
        result = np.zeros(numberOfValues, dtype='int32')
        performance_indices = np.zeros((numberOfValues, 3), dtype='double')
        
        if numberOfValues > 1:
            
            # evil hack: convert pandas data structure to an ordinary numpy
            # array and find out which indices are subject to change by calling
            # pandas and figure out what has changed.
            # We implicitly assume that no negative values are allowed.
            # Using custom-created numpy structures gains a tremendous speed-up
            # if the number of evaluation points is large. However, there is
            # a constant overhead of 6 accesses to the pandas dataframe.
            # Therefore, we avoid the hack altogether for single point
            # evaluations. The hack does not harm even for evaluating at merely
            # two points (6 vs. 6 dataframe accesses).
            weight_ref = dframe.loc[:,('weights_values', 'categorial_weight')].to_numpy(copy=True)
            dframe.loc[names_categories[0],('weights_values', 'categorial_weight')] = -42
            weight_aux = dframe.loc[:,('weights_values', 'categorial_weight')].to_numpy()

            iidx_0 = ~np.equal(weight_ref, weight_aux)

            weight_aux2 = dframe.loc[:,('weights_values', 'categorial_weight')].to_numpy(copy=True)
            dframe.loc[names_categories[1],('weights_values', 'categorial_weight')] = -84
            iidx_1 = ~np.equal(weight_aux2, weight_aux)
                
            # restore initial settings

            # blow the weight vector up to a matrix in order to use NumPy
            # vector operations and to avoid looping over single evaluation
            # points one after another
            weights_all_categories = np.outer(np.ones(numberOfValues), weight_ref)
            
            # replace the entries in the weight vector (aka the corresponding
            # rows in weights_all_categories) by the values in
            # weights_categories
            # np.outer computes the dyadic product of two vectors. This is
            # necessary here as we may replace more than one column per
            # criterion.
            weights_all_categories[:,iidx_0] = np.outer(weights_categories[:,0], np.ones(np.count_nonzero(iidx_0)))
            weights_all_categories[:,iidx_1] = np.outer(weights_categories[:,1], np.ones(np.count_nonzero(iidx_1)))
 
            # Scale weights_all_categories. This is not necessary for finding
            # the best performing alternatives, but for the correct
            # performance indices.
            if return_alternatives:
                # We face the problem, that the categorial weights occur several
                # times in weight_ref. We know, how often the weights for
                # names_categories[0] and names_categories[1] occur, but not,
                # how often the weights of the other categories occur.
                
                # Get all names of categories
                all_categories = dframe.index.levels[0].tolist()
                number_of_categories = len(all_categories)

                iidx = np.zeros((number_of_categories, weight_ref.shape[0]))
                iidx2 = np.zeros(number_of_categories, dtype='int32')

                for i in range(number_of_categories):
                    weight_ref = dframe.loc[:,('weights_values', 'categorial_weight')].to_numpy(copy=True)
                    dframe.loc[all_categories[i],('weights_values', 'categorial_weight')] = -1000
                    weight_aux = dframe.loc[:,('weights_values', 'categorial_weight')].to_numpy()

                    iidx[i,:] = ~np.equal(weight_ref, weight_aux)
                    iidx2[i] = np.where(iidx[i,] == 1)[0][0]
                    
                
                
                # get the index of the first entry being one
                sum_of_weights = weights_all_categories[:,iidx2].sum(axis=1)            

                weights_all_categories = weights_all_categories/sum_of_weights[:,np.newaxis]

            # blow weights_within_categories up to a matrix for NumPy-vectorization
            weights_within_categories = np.outer(np.ones(numberOfValues), weights_within_categories)

            # compute w_i * v_{i,j}
            weight_total = np.multiply(weights_all_categories, weights_within_categories) 
    
            #\sum_{i=1}^m w_i*(\sum_{j = 1}^{n(i)} v_{i,j} u_{i,j}^k ) is
            # acutally a matrix multiplication
            performance_indices = np.matmul(weight_total,option_values)
            result = performance_indices.argmax(1)
                
        else:
            for icategory in np.arange(len(names_categories)):
                dframe.loc[names_categories[icategory],('weights_values', 'categorial_weight')] = weights_categories[0,icategory]
        
            weights_all_categories = dframe.loc[:,('weights_values', 'categorial_weight')].to_numpy()

            # Scale weights_all_categories. This is not necessary for finding
            # the best performing alternative, but for the correct performance
            # indices.
            if return_alternatives:
 
                weight_ref = dframe.loc[:,('weights_values', 'categorial_weight')].to_numpy(copy=True)

               # Get all names of categories
                all_categories = dframe.index.levels[0].tolist()
                number_of_categories = len(all_categories)

                sum_of_weights = 0
                for i in range(number_of_categories):
                    sum_of_weights = sum_of_weights + dframe.loc[all_categories[i],('weights_values', 'categorial_weight')].values[0]
                
                weights_all_categories = weights_all_categories/sum_of_weights
            
            # total weighting factors w_i * v_{i,j} 
            weight_total = np.multiply(weights_all_categories, weights_within_categories)

            performance_indices[0,:] = np.dot(option_values.transpose(), weight_total)
            result[0] = performance_indices[0,:].argmax()

        # The result is a plain python type and not a NumPy type as Matlab can
        # deal with a list of python types but not with a list of NumPy types
        if (return_alternatives):
            return result.tolist(), performance_indices.tolist()
        else:
            return result.tolist()
        


    def get_class_and_scores_P2(self, names_categories, weights_categories, type_criterion, func_pars, return_alternatives):
        """
        In our investigations, we consider some categorial weights as
        variables. This function returns the result of the decision model aka
        the index of the most preferrable alternative for various weights given
        in weights_categories of selected categories provided in
        names_categories. The weights for all other categories are taken from
        the dframe_base-object.

        When called `get_class_at` will do a deepcopy of the
        `self.__dframe_base` containing the parameter set.
        The best performing alternative will then be calculated by PROMETHEE II.
        Algorithm:
        1. Build the decision matrix and set the values of the preference parameters and weights
        2. Calculate the deviations between the evaluations of the alternatives in each criteria
        3. Calculate the pairwise comparison matrix for each criterion
        4. Calculate the unicriterion net flows
        5. Calculate the weighted unicriterion flows
        6. Calculate the global preference net flows
        7. Rank the actions according to PROMETHEE II

        Parameters
        ----------
        names_categories : List of strings
            Names of categories considered variable
        weights_categories : numpy array containing the numerical values of the
            corresponding weighting factors (these are NOT the values of the
            criteria grouped in the categories!)
        return_alternatives: boolean
            if true, return the performance indices for all alternatives and
            the indices of the best performing alternatives, if false, return
            the indices of the alternatives only

        type_criterion: integer
            criterion following Brans et al., How to select and how to rank
            projects: The PROMETHEE method, European Journal of Operational
            Research 24 (1986) 228-238, Table 1
            1: P(x) = 1, x > 0; 0 elsewhere (piecewise constant)
            2: not yet implemented
            3: P(x) = cx, x>0; 0 elsewhere (linear, truncated to 1)
            4: not yet implemented
            5: not yet implemented
            6: P(x) = max(0, 1-exp(1/c^2 x^2)) (sigmoid function)

        func_par: list of doubles
            List of function parameters for the different criteria in
            Promethee II
            
            
        Returns
        -------
        integer list
            Indices of the best performing alternatives
            
        double list
            performance indices for any alternative for the weighting factors
            given in weights_categories (only for return_alternatives = True)
        """


        # Step 1
        dframe = self._dframe_base.copy(deep=True)
        option_values = dframe.pop('option_values').T.to_numpy()

        weights_within_categories = dframe.loc[:,('weights_values', 'subweight')].to_numpy()

        number_of_values = weights_categories[:,0].size
        number_of_alternatives = option_values.shape[0]
        number_of_criteria = weights_all_categories = len(dframe.loc[:,('weights_values', 'categorial_weight')].tolist())

       # Get all names of categories
        all_categories = dframe.index.levels[0].tolist()
        number_of_categories = len(all_categories)

        weights_all_categories = np.empty((number_of_values, number_of_criteria))

        if (number_of_values > 1):
            weight_ref = dframe.loc[:,('weights_values', 'categorial_weight')].to_numpy(copy=True)
            dframe.loc[names_categories[0],('weights_values', 'categorial_weight')] = -42
            weight_aux = dframe.loc[:,('weights_values', 'categorial_weight')].to_numpy()

            iidx_0 = ~np.equal(weight_ref, weight_aux)

            weight_aux2 = dframe.loc[:,('weights_values', 'categorial_weight')].to_numpy(copy=True)
            dframe.loc[names_categories[1],('weights_values', 'categorial_weight')] = -84
            iidx_1 = ~np.equal(weight_aux2, weight_aux)
                
            # restore initial settings

            # blow the weight vector up to a matrix in order to use NumPy
            # vector operations and to avoid looping over single evaluation
            # points one after another
            weights_all_categories = np.outer(np.ones(number_of_values), weight_ref)
            weights_all_categories[:,iidx_0] = np.outer(weights_categories[:,0], np.ones(np.count_nonzero(iidx_0)))
            weights_all_categories[:,iidx_1] = np.outer(weights_categories[:,1], np.ones(np.count_nonzero(iidx_1)))

            iidx = np.zeros((number_of_categories, weight_ref.shape[0]))
            iidx2 = np.zeros(number_of_categories, dtype='int32')

            for i in range(number_of_categories):
                weight_ref = dframe.loc[:,('weights_values', 'categorial_weight')].to_numpy(copy=True)
                dframe.loc[all_categories[i],('weights_values', 'categorial_weight')] = -1000
                weight_aux = dframe.loc[:,('weights_values', 'categorial_weight')].to_numpy()
                
                iidx[i,:] = ~np.equal(weight_ref, weight_aux)
                iidx2[i] = np.where(iidx[i,] == 1)[0][0]
                
                # get the index of the first entry being one
                sum_of_weights = weights_all_categories[:,iidx2].sum(axis=1)

        else:
            for icategory in np.arange(len(names_categories)):
                dframe.loc[names_categories[icategory],('weights_values', 'categorial_weight')] = weights_categories[0,icategory]
            
            weights_all_categories[0,:] = dframe.loc[:,('weights_values', 'categorial_weight')].to_numpy()        
           
            sum_of_weights = np.zeros(number_of_values)
            for i in range(number_of_categories):
                sum_of_weights[0] = sum_of_weights[0] + dframe.loc[all_categories[i],('weights_values', 'categorial_weight')].values[0]
        
        sum_of_weights = np.outer(sum_of_weights, np.ones(number_of_criteria))
        
        # normalize categorial weighting factors
        weights_all_categories = weights_all_categories/sum_of_weights

        del dframe  # save memory

        # total weighting factors w_i * v_{i,j} 
        weight_total = np.multiply(weights_all_categories, weights_within_categories)
        
        phi_net = np.zeros((number_of_values, number_of_alternatives))

        for criteria in range(number_of_criteria):

            # step 2
            uni = np.outer(option_values[:,criteria], np.ones(number_of_alternatives))
            uni = uni - uni.T

            # step 3
            if (type_criterion == 1):
                uni[uni>0] = 1
                uni[uni <=0.1] = 0
            elif (type_criterion == 3):
                uni = np.maximum(0, np.minimum(1, func_pars*uni))
            elif(type_criterion == 6):
                uni[uni <0] = 0
                uni = 1 - np.exp(-uni*uni/(func_pars**2))
            else:
                raise ModelError(f'Invalid choice of criterion inside Promethee II')

            # Step 4
            pos_flows = np.sum(uni, 1) / (number_of_alternatives - 1)
            neg_flows = np.sum(uni, 0) / (number_of_alternatives - 1)

            net_flows = pos_flows - neg_flows
                            
            # Step 5
            net_flows = np.outer(weight_total[:, criteria], net_flows)

            # Step 6
            phi_net = phi_net + net_flows

        if (return_alternatives):
            return phi_net.argmax(1).tolist(), phi_net.tolist()
        else:
            return phi_net.argmax(1).tolist()


    def normalize_within_categories(self, subweights):
        """
        This function returns the weighting factors v_{i,j} normalized with
        respect to categories aka it scales all subweights belonging to a
        category such that their sum is one.
        """

        index = subweights.index  # save Index to reapply it later

        # sum of subweights for each category
        sumsOfWeights = subweights.groupby(level=0, sort=False).agg(sum)

        # drop criteria index to merge on category-based index instead
        subweights = subweights.droplevel(axis=0, level=1)

        # BUG: sorts the resulting Series lexiographic no matter what
        # due to bug in merge
        # assign each subweight the correct sum
        merged = pd.merge(subweights, sumsOfWeights, left_index=True, right_index=True)
        
        with warnings.catch_warnings():
            # ignore warning raised by dividing by 0
            warnings.simplefilter('ignore')
            # divide each subweight by matching sum
            merged = merged.agg(lambda x: x[0]/x[1], axis=1)

        # fix series where the sum of subweights is 0
        merged.fillna(value=0, inplace=True)
        merged.index = index

        return merged


    def normalize_values(self, dframe):
        r"""
        This function enforces \sum_k |u_{i,j}^k| = 1 for the given DataFrame
        dframe.
        """

        column_index = dframe.columns

        options = dframe.pop('option_values') # TODO fix leak
        options = options.div(options.abs().sum(axis=1), axis=0, level=0)
        options.fillna(value=0, inplace=True)

        dframe = dframe.droplevel('lvl_1', axis=1).join(options,
                                                        on=options.index.names,
                                                        how='inner')
        dframe.columns = column_index

        return dframe


    def print_dframe(self):
        """Print the main DataFrame."""

        print(self._dframe_base.shape)

    def shape(self):
        dframe = self._dframe_base
        return (dframe.shape,
                dframe.columns)

    @property
    def dframe_base(self):
        """Getter for _dframe_base"""
        return self._dframe_base

    @dframe_base.setter
    def dframe_base(self, values):
        """Setter for _dframe_base"""
        # unbox values
        weight, performance = values
        expected_shape = (self._dframe_base['weights_values'].shape,
                 self._dframe_base['option_values'].shape)
        shape = (weight.shape, performance.shape)

        # test if given data shape matches existing shape
        if shape == expected_shape:
            # replace data chunks
            self._dframe_base.loc[:, 'weights_values'] = weight
            self._dframe_base.loc[:, 'option_values'] = performance
        else:
            raise ModelError(f'Expected data of shape {expected_shape} intstead of {shape}')

# --------------------------------Einlesen-------------------------------------

    def get_dataframe(self, file_name, sheet_name):
        """Read the main DataFrame from Excel and refine it.

        `get_dataframe` is called during the initialization of the model, when
        called it reads the parameter set contained in the specified Excel
        sheet and builds a `DataFrame` by:
            - Identifying index columns and assigning them as a `MultiIndex`.
            - Stripping empty columns and separating weight columns from value
              columns
            - Normalizing value columns
            

        Parameters
        ----------
        file_name : str
            Pathname to an Excel file (suffix .xlsx) containing the numerical
            values of the model parameters. The Excel file may consist of
            several worksheets.
        sheet_name : str
            Name of the Excel worksheet to read the numerical values of the
            model parameters from

        Returns
        -------
        dframe: DataFrame
            Refined `DataFrame` read from Excel.
        
        Raises
        ------
        ModelError
            Raised if the file structure does not corresponds to the expected
            format.
        """

        dframe = pd.read_excel(file_name, sheet_name=sheet_name, header=0)
        dframe = self._get_index_labels(dframe)
        dframe = self._get_column_labels(dframe)

        # enforce \sum_k |u_{i,j}^k| = 1.
        dframe = self.normalize_values(dframe)
        
        # prevent misalingnement of data due to sort bug in normalize_within_categories method
        dframe.sort_values(by='categories', axis=0, inplace=True)

        # enforce v_{i,1} + ... v_{i, n(i)} = 1 for all i
        dframe.loc[:,('weights_values', 'subweight')] = self.normalize_within_categories(dframe.loc[:,('weights_values', 'subweight')])
        return dframe

    def _get_index_labels(self, dframe):
        """Add Multiindex for the rows."""

        index_columns = []
        for column_nr, column in enumerate(dframe.columns):
            if not isinstance(dframe.iloc[0, column_nr], type(np.nan)):
                index_columns.append(column)
            else:
                break
        if len(index_columns) != 2:
            raise ModelError(f'''criteria columns should be 2 not
                             {len(index_columns)}''')
        else:
            dframe = dframe.set_index(index_columns)
            return dframe

    def _get_column_labels(self, dframe):
        """Add Multiindex for the columns."""

        seperator_columns = (dframe.drop(columns=dframe
                             .sum(axis=0, skipna=False).dropna().index)
                             .columns)
        columns = u.list_split(dframe.columns, seperator_columns)

        if len(columns[0]) != dframe.index.nlevels:
            raise ModelError('invalid input format')
        else:
            dframe.drop(columns=seperator_columns, inplace=True)
            dframe.columns = self._build_multiindex(columns)
            return dframe

    def _build_multiindex(self, grouped_labels):
        """Return Multiindex from two lists."""

        out = []
        top_level_labels = ['weights_values', 'option_values']

        if len(grouped_labels) != len(top_level_labels):
            raise ModelError('invalid input format')
        else:
            for super_label, labels in zip(top_level_labels, grouped_labels):
                for label in labels:
                    out.append((super_label, label))

            return pd.MultiIndex.from_tuples(out, names=('lvl_1', 'lvl_2'))


# -----------------------------------------------------------------------------


class ModelError(Exception):
    """Exception raised if model parameters format is invalid"""


def main():
    """Test Data class."""
    data = Data('Modellparameter.xlsx', 'car_users')
    data.print_dframe()


if __name__ == '__main__':
    main()
