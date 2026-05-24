This repository houses the Python scripts and Jupyter Notebooks for a serverless AQI prediction application built
for the completion of my internship at 10Pearls Pakistan.

The application fetches weather + AQI data from OpenMeteo for the past 3 months as a historical data backfill, performs feature
engineering, and stores the data in a Hopsworks Feature Store. The application also automates hourly data gathering using GitHub Actions.
The stored data is fetched from the Feature Store every 24 hours (also automated), and five machine learning models are trained on it.
The trained models are then deployed to Hopsworks after the previously stored and deployed models are removed.

The frontend is designed in Streamlit and is deployed at: 

https://haroons-pearls-aqi-predictor.streamlit.app/
