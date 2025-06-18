# Team Performance Neural Network

## Overview
A lightweight feed-forward neural network that predicts football game outcomes by analyzing each team’s recent performance. It captures momentum and situational context to deliver fast, data-driven insights.

## Key Features
- **Sliding 3-Game Windows:** Short-term fluctuations by aggregating stats over the last three games.
- **Red Zone Efficiency:** Measures scoring success rate inside the 20-yard line.
- **Yards per Point Margin:** Reflects scoring efficiency by dividing total yards by point differential.
- **Plays per Point Margin:** Indicates scoring pace by dividing total plays by point differential.
- **Division Game Flag:** Highlights intra-division matchups for rivalry context.
- **Weather Adjustments:** Incorporates temperature, precipitation, and wind speed to account for environmental effects.

## K-fold (stratified) early stoppage
- Determining early stopping using Keras’s built-in callback on a single validation split led to inconsistent results: the training and test errors averaged around 0.22%, while the validation error was about 0.30%. This suggests some inherent noise in the validation set. To mitigate this, switched to a stratified k-fold cross-validation for early stopping instead of a single hold-out split.
