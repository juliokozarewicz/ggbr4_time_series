# **DESCRIPTION**

## Study of Gerdau's share price using time series and also the production of cars and the volume traded as exogenous variables. To predict the stock price, it was first necessary to make a forecast, also through time series, for the production of cars and the volume traded.

Model results for predicting car production: https://github.com/juliokozarewicz/car_production

>This is not a buy or sell recommendation, it is just a study and should not be considered when making an investment decision!

## Car Production
Values of the production of motor vehicles in the country. They reflect the sales performance of companies associated with:

• Stock and sale of vehicles by dealerships;

• Production and sales of vehicles and similar;

• Motorcycle production and sales.

## GERDAU S.A.
Gerdau is the largest producer of long steel in Latin America, with steel mills in Brazil, Argentina, Canada, Colombia, Dominican Republic, Mexico, Peru, United States, Uruguay and Venezuela. Currently, Gerdau has an installed capacity of 26 million metric tons of steel per year and offers steel for the civil construction, automobile, industrial, agricultural and various sectors. Gerdau is also the world’s 30th largest steelmaker. It has 337 industrial and commercial units and more than 30,000 employees across 10 countries. Gerdau produces long carbon steel, long special steel, flat steel and forged and cast parts. These products are used in different sectors, such as industry, metallurgy, farming and livestock, civil construction, automotive industries, petrochemicals, railway and naval sectors, in addition to orthodontic, medical and food areas. Gerdau is also the main supplier of specialty steel for the international automotive network.

Data source: Yahoo finance.

# **RESULTS**
Results obtained through the model estimation process.

## Variable analysis at level:
<img src="2_results/1_time_serie.jpg"> <br /> <br />
<img src="2_results/2_fac_facp_level.jpg"> <br /> <br />
<img src="2_results/3_periodogram_level.jpg"> <br /> <br />
<img src="2_results/4.jpg"> <br /> <br />

## SEASONAL ADJUSTMENT:
<img src="2_results/5_x13_results.jpg"> <br /> <br />
<img src="2_results/6_x13_seasonal_adjustment.jpg"> <br /> <br />

## Study of data stationarity:
<img src="2_results/7.jpg"> <br /> <br />
<img src="2_results/8.jpg"> <br /> <br />


## Model results:
<img src="2_results/9.jpg"> <br /> <br />

## Residual analysis:
<img src="2_results/10_residuals (acf and pacf).jpg"> <br /> <br />
<img src="2_results/11_residuals (frequency distribution).jpg"> <br /> <br />
<img src="2_results/12_residuals (time serie).jpg"> <br /> <br />

# MODEL TEST:
<img src="2_results/13_observed_fitted_predict_test.jpg"> <br /> <br />

# FORECAST:
<img src="2_results/13_observed_fitted_predict.jpg"> <br /> <br />
