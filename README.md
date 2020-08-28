# SF_EMS_Call_Rate_Prediction

### Table of Contents
You're sections headers will be used to reference location of destination.

- [Description](#description)
- [Data Overview](#data-overview)
- [Data Exploration](#data-exploration)
- [Models](#models)
- [Takeaway](#takeaway)

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

-Separated data by neighborhood of event

-Aggregated data in 3H time periods

#### EDA - Seasonality across 20 years
![Yearly Trend](/Photos/Yearly_Neigh.png)

![Monthly Seasonality](/Photos/Monthly_Neigh.png)

![Weekly Seasonality](/Photos/Weekly_Neigh.png)

![Daily Seasonality](/Photos/Daily_Neigh.png)

##### ^All time measurements seemed to match up similarly, so used the total time (call to scene)

[Back To The Top](#read-me-template)

---

## Models
### First
**Null Hypothesis:** Areas with high homeless population have same response rate as areas with low homeless population.

**Alternative Hypothesis:** Areas with high homeless population have different response rate as areas with low homeless population.

**P-Value** of 0.02 for rejection

### Second
I wanted to investigate my data a bit further, so I decided to break up the data into calls that were life-threatening and calls that were not life threatening. I did not perform EDA on this data but used it in another hypothesis test.

**Null Hypothesis:** Calls marked as life-threatening have same response rate as calls marked as non-life threatening.

**Alternative Hypothesis:** Calls marked as life-threatening have different response rate as calls marked as non-life threatening.

**P-Value** of 0.02 for rejection


[Back To The Top](#read-me-template)

---

## Results
### First
High Pop Avg: 530 seconds

Low Pop Avg: 516 seconds

P-Value: 0.0 (very very low)

**Reject Null Hypothesis**

![Homeless Hist](/High_to_Low_Homeless_Hist.png)

### Second
Life Threatening Avg: 508 seconds

Non-Life Threatening Avg: 528 seconds

P-Value: 4 e-214

**Reject Null Hypothesis**

![Homeless Hist](/Life_Threat_to_Non_Hist.png)

## Takeaway
The data suggests with very high confidence that the areas with low homeless populations in San Francisco have a slightly faster EMS response time, and that calls marked as life-threatening have a faster response time. The high confidence is not surprising as this dataset included all EMS calls in San Francisco, so was the population data. There is not suggestion of causation in this analysis.

[Back To The Top]

