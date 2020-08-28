# SF_EMS_Call_Rate_Prediction

### Table of Contents

- [Description](#description)
- [Data Overview](#data-overview)
- [Data Exploration](#data-exploration)
- [Models](#models)
- [By Neighborhood](#by-neighborhood)
- [Next Steps](#next-steps)

---

## Description

One of the most significant factors in surviving a medial emergency is the response time of EMS. By forecasting EMS call by time of day for the different neighborhoods in a city, the distribution of EMS units can be optimized to prepare for the the fluctuations in call rates/episodes. Having units closer to being in the right place, right time would naturally reduce average response rates and optimize the resources available.

Aggregating data on a larger scale (day/week/month) can also be used to identify and predict causes for high EMS call rates and prepare emergency services ahead of time.

## Data Overview

All EMS calls through SFFD available through [sfgov.org](https://data.sfgov.org/Public-Safety/Fire-Department-Calls-for-Service/nuek-vuh3)
 - ~2.6 Million unique EMS calls from 2000 to 2020
 - Timestamp inclues year,month,day,hour,minute,second
 


[Back To The Top](#read-me-template)

---

## Data Exploration

#### Cleaned Data
-Ended with ~2.6 M rows of useable EMS calls spanning April 2000 to present (July 2020)

-Separated data by neighborhood of event (but left combined for model selection)

-Aggregated data in 3H time periods

#### EDA - Seasonality across 20 years
![Yearly Trend](/Photos/Yearly_Neigh.png)

![Monthly Seasonality](/Photos/Monthly_Neigh.png)

![Weekly Seasonality](/Photos/Weekly_Neigh.png)

![Daily Seasonality](/Photos/Daily_Neigh.png)

[Back To The Top](#read-me-template)

---

## Models (All Neighborhoods Combined)
### SARMIAX

Used a ARIMA model with a seasonality feature to capture daily seasonality of data.

This model proved to be computationally expensive with the amount of data I was working with. Additionally, since SARIMAX does not handle multi-seasonality well, I decided to run the model using 3 weeks to predict a 4th week.

**Setup:** Picked initial hyperparameters using correlation/autocorrelation and then performed a grid search and chose model with lowest RMSE

![SARIMAX Model 1 Week Prediction](/Photos/ARIMA_pred.png)
#### Second half of model is test data, first half is train data

**RSME**  = 7.4

### Prophet

Used a Prophet model to capture multi-seasonality in data.

This model was much less computationally expensive. I limited my data to a 2 year stretch and and trained the model on the full two years up until the week to be forecast.

**Setup:** Used default parameters for Prophet model

![Prophet Model 1 Week Prediction](/Photos/Proph_pred_week.png)
#### Second half of model is test data, first half is train data

**RSME**  = 7.2

### Control

Used the previous week to predict the current week

**RSME**  = 8.8

[Back To The Top](#read-me-template)

---

## By Neighborhood
### Best Model
The Model that I selected for further analysis was the Prophet model because it scored higher

### Daily (2 week period)

![Daily Rates by Neighborhood](/Photos/3H_Daily_Neigh.png)

### Weekly

![Weekly Rates by Neighborhood](/Photos/Daily_Pred_Neigh.png)


## Next Steps
-The models should be tested on larger sets of data to confirm accuracy. 

-The available parameters for both SARIMAX and Prophet can explored and optimized further to capture higher level seasonality better

-The models can be expanded to be utilized for anomaly detection, which can be used to explore non-seasonal feature that may be good indicators of EMS call rates.

-An LSTM model can be trained to take in features of the data (like call type/severity/weather/etc) which may build a stronger model.

[Back To The Top]

