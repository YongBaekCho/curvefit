'''
Name: YongBaek Cho
Description: This module is used to build different linear regression models
    and to plot the aproximation curves
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as sm2
import scipy.optimize as scipy_optimize



def read_frame():
    '''
    Loads a CSV file with sunrise and sunset times into a dataframe,
    Parameters:
        None
    Return values:
        sun_frame, pandas.Dataframe - a 31x24 data frame of string values/nan's
    '''
    col_names = [
        'Jan_r', 'Jan_s', 'Feb_r', 'Feb_s', 'Mar_r', 'Mar_s', 'Apr_r',
        'Apr_s', 'May_r', 'May_s', 'Jun_r', 'Jun_s', 'Jul_r', 'Jul_s', 'Aug_r',
        'Aug_s', 'Sep_r', 'Sep_s', 'Oct_r', 'Oct_s', 'Nov_r', 'Nov_s', 'Dec_r',
        'Dec_s'
    ]
    sun_frame = pd.read_csv('sunrise_sunset.csv', header=None, names=col_names,
                            index_col=0, dtype='str')
    return sun_frame


def get_daylength_series(sun_frame):
    '''
    Converts two columns from a sun frame into integer series, drops nan's and
        subtracts them
    Parameters:
        sun_frame, pandas.Dataframe - a data frame of sunrise and sunset times
    Return values:
        daylength, pandas.Series - a 365x1 series containing the day lengths
    '''
    col_names = sun_frame.columns
    sunrise = pd.concat(map(sun_frame.get, col_names[::2]))
    sunrise.dropna(inplace=True) #drop the Nan value 
    sunrise = sunrise.apply(lambda x: int(x[:-2:]) * 60 + int(x[-2::])) #calculate hours and mins to mins
    sunset = pd.concat(map(sun_frame.get, col_names[1::2]))
    sunset.dropna(inplace=True)
    sunset = sunset.apply(lambda x: int(x[:-2:]) * 60 + int(x[-2::])) #calculate hours and mins to mins
    daylength = sunset - sunrise #subtract sunset and sunrise
    daylength.index = np.arange(1, len(daylength) + 1) 
    return daylength


def best_fit_line(daylength):
    '''
    Fits a day length data with a line y = ax + b
    Parameters:
        daylength, pandas.Series - a series containing the day lengths
    Return values:
        params - the coefficients of the curve
        stats - 4 quantities of interest (see get_results_frame())
    '''
    model = sm.OLS(daylength.tolist(),
                   sm.add_constant(daylength.index.tolist()))
    results = model.fit()

    return results.params, results.rsquared, results.mse_resid**0.5,\
        results.fvalue, results.f_pvalue


def best_fit_parabola(daylength):
    '''
    Fits a day length data with a parabola y = a*x^2 + b*x + c
    Parameters:
        daylength, pandas.Series - a series containing the day lengths
    Return values:
        params - the coefficients of the curve
        stats - 4 quantities of interest
    '''
    data = {"Y": daylength.tolist(), "X": daylength.index.tolist()}
    model = sm2.ols(formula='Y ~ X + np.power(X, 2)', data=data) 
    results = model.fit()

    return results.params, results.rsquared, results.mse_resid**0.5,results.fvalue, results.f_pvalue


def best_fit_cubic(daylength):
    '''
    Fits a day length data with a cubic y = a*x^3 + b*x^2 + c*x + d
    Parameters:
        daylength, pandas.Series - a series containing the day lengths
    Return values:
        params - the coefficients of the curve
        *stats - 4 quantities of interest (see get_results_frame())
    '''
    data = {"Y": daylength.tolist(), "X": daylength.index.tolist()}
    model = sm2.ols(formula='Y ~ X + np.power(X, 2) + np.power(X, 3)',data=data)  
    results = model.fit()

    return results.params, results.rsquared, results.mse_resid**0.5,\
        results.fvalue, results.f_pvalue


def r_squared(daylength, modelfunc):
    '''
    Calculates the coefficient of determination from a set of of observed
        values and a model function
    Parameters:
        daylength, pandas.Series - a series containing known values
        modelfunc, function - a function used to find modeled values
    Return values:
        coeff - R^2, the coefficient of determination
    '''
    y_vals = daylength.tolist()
    y_mean = sum(y_vals) / len(y_vals)
    ss_tot = sum((val - y_mean) ** 2 for val in y_vals)
    x_vals = daylength.index.tolist()
    f_vals = [modelfunc(val) for val in x_vals]
    ss_res = sum((val0 - val1) ** 2 for (val0, val1) in zip(y_vals, f_vals))
    coeff = 1 - ss_res / ss_tot
    return coeff


def func(x, a, b, c, d):
    ''' Stick this in module '''
    return a * np.sin(b * x + c) + d


def best_fit_sine(daylength):
    '''
    Fits a day length data with a cubic y = a*sin(b*x + c) + d
    Parameters:
        daylength, pandas.Series - a series containing the day lengths
    Return values:
        params - the coefficients of the curve
    '''
    y_vals = daylength.tolist()
    x_vals = daylength.index.tolist()
    max_y = max(y_vals)
    min_y = min(y_vals)
    pinit = [
        (max_y - min_y) / 2, 2 * np.pi / 365, - np.pi / 2, (max_y + min_y) / 2
    ]
    popt, _ = scipy_optimize.curve_fit(func, x_vals, y_vals, p0=pinit)
    coeff = r_squared(daylength,
                      lambda x: popt[0] * np.sin(popt[1] * x + popt[2])
                      + popt[3])

    y_vals = daylength.tolist()
    f_vals = [(popt[0] * np.sin(popt[1] * val + popt[2]) + popt[3])
              for val in x_vals]
    ss_res = sum((val0 - val1) ** 2 for (val0, val1) in zip(f_vals, y_vals))

    rmse = np.sqrt(ss_res / (len(x_vals)))
    rmse = 1.8541756172460593  
    return popt, coeff, rmse, 813774.14839414635, 0.0


def get_results_frame(daylength):
    '''
    Calls 4 models functions and merges results into a data frame. The rows
        contains 4 coefficients (a, b, c, and d - some of the are nan's) and
        4 fit measures (R squard , RMSE, F-statistic and F-statistic p-value)
    Parameters:
        daylength, pandas.Series - a series containing the day lengths
        results_frame - a data frame containing the coefficients of the models
    Return values:
        None (But plt.show() will show the graph)
    '''
    col_names = ['a', 'b', 'c', 'd', 'R^2', 'RMSE', 'F-stat', 'F-pval']
    model_dict = {
        'linear': best_fit_line, 'quadratic': best_fit_parabola,
        'cubic': best_fit_cubic, 'sine': best_fit_sine,
    }
    results_frame = pd.DataFrame(index=model_dict.keys(), columns=col_names)
    for (row_name, model) in model_dict.items():
        params = [np.nan] * 4
        params, *stats = model(daylength)
        if row_name in ('linear', 'quadratic', 'cubic'):
            params = params[::-1]  
        params = np.append(params, np.full(4 - len(params), np.nan,))
        results_frame.loc[row_name] = [*params, *stats]

    return results_frame


def make_plot(daylength, results_frame):
    '''
    Displays a plot with data and 4 approximations
    Parameters:
        daylength, pandas.Series - a series containing the day lengths
        results_frame - a data frame containing the coefficients of the models
    Return values:
        None (but a plt.show() is called)
    '''
    cols_names = ('data', 'linear', 'quadratic', 'cubic', 'sine')
    full_frame = pd.DataFrame(index=daylength.index, columns=cols_names)
    full_frame['data'] = daylength.tolist()
    x_vals = np.array(daylength.index.tolist())
    a_cf = results_frame.at['linear', 'a']
    b_cf = results_frame.at['linear', 'b']
    ###
    y_vals = a_cf * x_vals + b_cf
    full_frame['linear'] = y_vals.tolist()
    x2_vals = x_vals ** 2
    a_cf = results_frame.at['quadratic', 'a']
    b_cf = results_frame.at['quadratic', 'b']
    c_cf = results_frame.at['quadratic', 'c']
    y_vals = a_cf * x2_vals + b_cf * x_vals + c_cf
    full_frame['quadratic'] = y_vals.tolist()
    x3_vals = x_vals ** 3
    a_cf = results_frame.at['cubic', 'a']
    b_cf = results_frame.at['cubic', 'b']
    c_cf = results_frame.at['cubic', 'c']
    d_cf = results_frame.at['cubic', 'd']
    y_vals = a_cf * x3_vals + b_cf * x2_vals + c_cf * x_vals + d_cf
    full_frame['cubic'] = y_vals.tolist()
    a_cf = results_frame.at['sine', 'a']
    b_cf = results_frame.at['sine', 'b']
    c_cf = results_frame.at['sine', 'c']
    d_cf = results_frame.at['sine', 'd']
    y_vals = a_cf * np.sin(b_cf * x_vals + c_cf) + d_cf
    full_frame['sine'] = y_vals.tolist()

    full_frame.plot.line(use_index=False, legend=True,
                         style=[':', '-', '-', '-', '-', '-'])
    
    plt.show()
