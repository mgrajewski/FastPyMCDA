# -*- coding: utf-8 -*-
"""
This class is a wrapper class for the decision model implemented in 'model.py'.
For details regarding the model itself, we refer to the documentation in
'model.py'.
In our investigations, we concentrate on the influence of the weighting factors
of categories on the decisions made. To visualise that dependency, we
consider two of these categorial weighting factors variable and the others
fixed. Then, any combination of concrete values for the two variable
weighting factors for the given categories can be considered as a point in the
plane, and we colour that point according to the alternative chosen. The
resulting image may serve as a starting point for further analysis. Moreover,
we visualise the performance indices of the different alternatives as well in
order to assess the robustness of a decision.

This approach is reflected in the implementation of the decision models and
their wrapper class. For the sake of flexibility, we do not hard-code the
model parameters (consisting of weights and values/characteristics), but read
the corresponding numerical values from an Excel file when creating a new
instance of the wrapper class. We store these parameters using pandas in the
wrapper object.

It turned out that reading from Excel and manipulating pandas-DataFrames is
rather time-consuming. When just running the model once, the runtime is
neglegible, but e.g. for stochastic analysis, we need to run the model up to
some million times. Then, runtime can be an issue. Therefore, we read the
Excel file only once during initialisation. We store the model parameters in a
DataFrame-object for convenience, but implemented the core of the model using
NumPy. Reading the DataFrame happens once per call of get_class_and_scores and
the related functions. Thanks to this, it is way faster to call
get_class_and_scores once for a large set of weighting factors than calling it
many times for every single combination of weighting factors.

@author: Simon Grasse, Forschungszentrum Jülich
Matthias Grajewski, FH Aachen University of Applied Sciences
Stefan Vögele, Forschungszentrum Jülich
"""

import numpy as np
import sys
import os

# make sure root package-directory is in python search-path
if '__file__' in globals():
    ABS_DIR_NAME = os.path.abspath(os.path.dirname(__file__))
else:
    ABS_DIR_NAME = os.getcwd()

if os.path.abspath(os.path.join(ABS_DIR_NAME, '..', '..')) not in sys.path:
    sys.path.append(os.path.abspath(os.path.join(ABS_DIR_NAME, '..', '..')))

from pymcda.src import model

class Wrapper():
    """Evaluates the model for a set of categorial weights.

    When called, `Wrapper` initializes an instance of the class `model.Data`.
    """

    def __init__(self, file_name, sheet_name):

        """
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

        self.data = model.Data(file_name, sheet_name)


    def set_weights_of_categories(self, names_categories, weights_categories):
        """
        This function sets weighting factors for selected categories. The changes
        apply to the data-object within the current instance and are therefore
        global and permanent (up to a renewed call of this function).

        Parameters
        ----------
        names_categories : list of strings
            Names of the categories whose weight factors are to be changed.            
        weights_categories : list of lists containing the numerical values for
            the weight factors belonging to the categories defined in
            names_categories.

        Returns
        -------
            -
        """
        number_of_categories = len(names_categories)
        if (number_of_categories != len(weights_categories)):
            raise WrapperError(f'The number of categories and the number of weights to assign differ.')
        
        for icategory in np.arange(number_of_categories):
            self.data.dframe_base.loc[names_categories[icategory],('weights_values', 'categorial_weight')] = weights_categories[icategory]
            
        # We do not have to ensure here that the sum of weighting factors of
        # the categories remains one as this does not affect the decision
        # itself. However, it does affect the performance indices, but in this
        # case, we divide in model.get_class_and_scores and the related
        # functions by the sum of the weighting factors of the categories
        # anyway.
            
    def set_values_of_criteria(self, names_categories, names_criteria, values_criteria):
        """
        This function sets new values for selected criteria in the same or 
        belonging to different categories. This is done for all alternatives
        simultaneously. The changes apply to the data-object within the
        current instance and are therefore global and permanent (up to a
        renewed call of this function). The criteria are in the
        dataframe-object grouped by categories which must be provided as well.

        Parameters
        ----------
        names_categories : list of strings
            Names of the categories the criteria are assigned to.            
        names_criteria : list of strings
            Names of the criteria within the categories.            
        values_criteria : list of lists containing the numerical values for
            the values belonging to the criteria defined in names_criteria.

        Returns
        -------
            -
        """
        number_of_criteria = len(names_criteria)
        if (number_of_criteria != len(names_categories)):
            raise WrapperError(f'The number of categories and the number of criteria differ.')
        
        if (number_of_criteria != len(values_criteria)):
            raise WrapperError(f'The number of criteria and the number of values to assign differ.')
        
        for icriterion in np.arange(number_of_criteria):
            self.data.dframe_base.loc[(names_categories[icriterion], names_criteria[icriterion]), ('option_values')] = values_criteria[icriterion]

        # enforce \sum_k u_{i,j}^k = 1 in the next lines of code

        dframe = self.data.dframe_base

        # get relevant part of the data frame
        aux = dframe[('option_values')]

        # divide by absolute row sum
        aux = aux.div(aux.abs().sum(axis=1), axis=0, level=0)

        # replace NaNs by 0
        aux.fillna(value=0, inplace=True)
        
        # replace relevant part of the data frame by the scaled version
        dframe[('option_values')] = aux
        
        
    def get_class_for_weights(self, category1_name, category2_name, weights, *add_args):
        """
        Evaluates the decision model for a set of weighting factors of the two
        categories named 'category1_name' and 'category2_name'.

        When called, `get_class_for_weights` processes `weights` into a
        NumPy-array, which is then passed to `self.data.get_class_and_scores'
        in order to evaluate the indices of the alternatives chosen for the
        different values of the weighting factors for categories
        `category1_name` and `category2_name`.

        Parameters
        ----------
        category1_name : String
            Name of the first category.
        category2_name : String
            Name of the second category.
        weights : tuple of two lists containing the numerical values for the
            weighting factors belonging to categories category1_name and
            category2_name.
        *add_args: List
            additional arguments for the decision model. Arguments must
            provided in this order:
            - type of model (0: extended SAW, 1: Promethee II)
            - additional parameters for Promethee II: type of criterion
             (integer), shape parameter(s) if necessary (single value as double,
             if more values are necessary, provide a list)

        Returns
        -------
            list with indices of the alternatives chosen

        Raises
        ------
            WrapperError
        """

        # if there are no additional arguments, we take extended SAW, which
        # does not need any additional parameters
        if (not add_args):
            model_type = 0
        else:
            model_type = add_args[0]
                
        # perform consistency checks
        if (model_type == 1):
            # we need at least one additional input: the type of criterion
            if (len(add_args)>=2):
                type_criterion = add_args[1]
            else:
                raise WrapperError('Additional arguments for model specification are missing.')    
            
            # these type of criterion need additional parameters
            if (type_criterion in [3,6]):
                if (len(add_args)>=3):
                    func_pars = add_args[2]
                else:
                    raise WrapperError('One or more shape parameters for Promethee II are missing.')    


        categories_name = (category1_name, category2_name)
        if (category1_name == ''):
            raise WrapperError(f'no name provided for category 1')

        if (category2_name == ''):
            raise WrapperError(f'no name provided for category 2')
            
        if len(weights) != 2:
            raise WrapperError('weights must consist of two lists with weighting factors')
            
        if not(category1_name in self.data.dframe_base.index):
            raise WrapperError(f'A category named {category1_name} does not exist.')

        if not(category2_name in self.data.dframe_base.index):
            raise WrapperError(f'A category named {category2_name} does not exist.')


        # convert the input into numpy arrays for the sake of performance
        weights0 = np.array(weights[0])
        weights1 = np.array(weights[1])

        if (weights0.size != weights1.size):
            raise WrapperError('''Equal number of weighting factors for each category
                               expected''')
            
        NumPy_weights = np.column_stack((weights0, weights1))

        # extended SAW
        if (model_type == 0):
            classes = self.data.get_class_and_scores(categories_name, NumPy_weights, return_alternatives=False)

        # Promethee II
        elif(model_type == 1):
            classes = self.data.get_class_and_scores_P2(categories_name, NumPy_weights, type_criterion, func_pars, return_alternatives=False)
        else:
            raise WrapperError(f'Invalid choice of model')
        
        return(classes)


    def get_class_and_scores_for_weights(self, category1_name, category2_name, weights, *add_args):
        """
        Evaluates the decision model for a set of weighting factors of the two
        categories named 'category1_name' and 'category2_name'.

        When called, `get_class_for_weights` processes `weights` into a
        NumPy-array, which is then passed to `self.data.get_class_and_scores'
        in order to evaluate the indices of the alternatives chosen for the
        different values of the weighting factors for categories
        `category1_name` and `category2_name`. Moreover, the function
        provides the performance indices for all alternatives.

        Parameters
        ----------
        category1_name : String
            Name of first category.
        category2_name : String
            Name of second category.
        weights : tuple of two lists containing the numerical values for the
            weighting factors belonging to categories category1_name and
            category2_name.
            0: extended SAW (default, if model_type is not provided)
            1: Promethee II
        *add_args: List
            additional arguments for the decision model. Arguments must
            provided in this order:
            - type of model (0: extended SAW, 1: Promethee II)
            - additional parameters for Promethee II: type of criterion
             (integer), shape parameter(s) if necessary (single value as double,
             if more values are necessary, provide a list)


        Returns
        -------
            list with indices of the alternatives chosen
            list with performance indices of all alternatives

        Raises
        ------
            WrapperError
        """
        
        # if there are no additional arguments, we take extended SAW, which
        # does not need any additional parameters
        if (not add_args):
            model_type = 0
        else:
            model_type = add_args[0]
                
        # perform consistency checks
        if (model_type == 1):
            # we need at least one additional input: the type of criterion
            if (len(add_args)>=2):
                type_criterion = add_args[1]
            else:
                raise WrapperError('Additional arguments for model specification are missing.')    
            
            # these type of criterion need additional parameters
            if (type_criterion in [3,6]):
                if (len(add_args)>=3):
                    func_pars = add_args[2]
                else:
                    raise WrapperError('One or more shape parameters for Promethee II are missing.')    
           

        categories_name = (category1_name, category2_name)
        if (category1_name == ''):
            raise WrapperError(f'no name provided for category 1')

        if (category2_name == ''):
            raise WrapperError(f'no name provided for weight 2')
            
        if len(weights) != 2:
            raise WrapperError('weights must consist of two lists with weighting factors')
            
        if not(category1_name in self.data.dframe_base.index):
            raise WrapperError(f'A category named {category1_name} does not exist.')

        if not(category2_name in self.data.dframe_base.index):
            raise WrapperError(f'A category named {category2_name} does not exist.')


        # convert the input into numpy arrays for the sake of performance
        weights0 = np.array(weights[0])
        weights1 = np.array(weights[1])

        if (weights0.size != weights1.size):
            raise WrapperError('''Equal number of weighting factors for each category
                               expected''')
        
        NumPy_weights = np.column_stack((weights0, weights1))

        # extended SAW
        if (model_type == 0):
            classes, scores = self.data.get_class_and_scores(categories_name, NumPy_weights, return_alternatives=True)

        # Promethee II
        elif (model_type == 1):            
            classes, scores = self.data.get_class_and_scores_P2(categories_name, NumPy_weights, type_criterion, func_pars, return_alternatives=True)
        else:
            raise WrapperError(f'Invalid choice of model')

        return(classes, scores)



def main():
    """Test method `wrapper`."""
    from pymcda.test.test_wrapper import test
    print('loades as __main__')
    test()



class WrapperError(Exception):
    """Exception raised if wrapper obtains invalid arguments."""


if __name__ == '__main__':
    main()
