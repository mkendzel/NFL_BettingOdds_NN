# Team Performance Neural Network

## Overview
A lightweight feed-forward neural network that predicts football game outcomes by analyzing each team’s recent performance. It captures momentum and situational context to deliver fast, data-driven insights.

## Key Features
- **Sliding 3-Game Window:** Smooths short-term fluctuations by aggregating stats over the last three games.
- **Red Zone Efficiency:** Measures scoring success rate inside the 20-yard line.
- **Yards per Point Margin:** Reflects scoring efficiency by dividing total yards by point differential.
- **Plays per Point Margin:** Indicates scoring pace by dividing total plays by point differential.
- **Division Game Flag:** Highlights intra-division matchups for rivalry context.
- **Weather Adjustments:** Incorporates temperature, precipitation, and wind speed to account for environmental effects.

## K-fold (stratified) early stoppage
- Using the validation set to determine early stoppage via Keras's built in callback function as done in the default NN produced discrepancies in accuracy score. On average, the train and test errors were about 0.22% while the validation set was about 0.30%. This coudl be an issue of over-tuning to the validation set and its unique idiosyncrasies and its noise within. Thus, I used a stratified k-fold cross-validation for early stoppage rather than the single val split
