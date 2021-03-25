from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import pandas as pd
import numpy as np

  # Gathering data
boston_dataset= load_boston()
data= pd.DataFrame(data= boston_dataset.data, columns= boston_dataset.feature_names)
features= data.drop(['INDUS','AGE'], axis= 1)

log_prices= np.log(boston_dataset.target)
target= pd.DataFrame(data= log_prices, columns= ['PRICE'])

RM_IDX= 4
RAD_IDX= 6
PT_RATIO= 8
CHAS_IDX=2
B_IDX= 10

ZILLOW_MEDIAN_PRICE= 659.8
SCALE_FACTOR= ZILLOW_MEDIAN_PRICE / np.median(boston_dataset.target)

property_stats= features.mean().values.reshape(1,11)

regr= LinearRegression()
regr.fit(features, target)
fitted_vals= regr.predict(features)

MSE=  mean_squared_error(target, fitted_vals)
RMSE= np.sqrt(MSE)


def get_log_estimate(nr_rooms, students_per_classrooms, proportion_of_blacks, next_to_river=False, high_confidence=True):
    
    #confugure
    property_stats[0][RM_IDX]= nr_rooms
    property_stats[0][PT_RATIO]= students_per_classrooms
    property_stats[0][B_IDX]= proportion_of_blacks
    
    if next_to_river:
        property_stats[0][CHAS_IDX]= 1
        
    else:
        property_stats[0][CHAS_IDX]= 0
    
    
    # make predictions
    
    log_estimate= regr.predict(property_stats)[0][0]
    
    # calc range
    
    if high_confidence:
        upper_bound= log_estimate + 2*RMSE
        lower_bound= log_estimate - 2*RMSE
        interval= 95
        
    else:
        upper_bound= log_estimate + RMSE
        lower_bound= log_estimate - RMSE
        interval= 68
        
        return log_estimate, upper_bound, lower_bound, interval
    
    
def get_dollar_estimate(rm, ptratio, black_prop, chas= False, large_range=False ):

    """ Estimate house prices in boston.
    
    Keyword arguments.
    
    rm- number of rooms in the property (rm cannot be less than one).
    ptratio- the number of students per teacher in the classroom (ptratio cannot be less than one).
    black_prop- the proportion of blacks in the area the property is situated.
    chas- True if property is next to the river, False Otherwise
    large_range- True for a 95% confidence, False for a 68% confidence
     
    
    """
    
    if rm < 1 or ptratio < 1:
        print(' this is unrealistic. Try again')
        return
    
    log_est, upper, lower, conf = get_log_estimate( nr_rooms= rm , students_per_classrooms= ptratio, proportion_of_blacks= black_prop,
                                        
                                                         next_to_river=chas, high_confidence= large_range)
          
    # convert to todays dollar value


    dollar_est= np.e**log_est * 1000 * SCALE_FACTOR
    dollar_hi= np.e**upper * 1000 * SCALE_FACTOR
    dollar_low= np.e**lower * 1000 * SCALE_FACTOR

    # round the values to todays dollars

    round_est= np.around(dollar_est, -3)
    round_hi= np.around(dollar_hi, -3)
    round_low= np.around(dollar_low, -3)


    print(f'the estimated value of the property is, {round_est}.')
    print(f'at {conf}% confidence, the valuation is,')
    print(f'USD low is {round_low}, USD high is {round_hi}.')
