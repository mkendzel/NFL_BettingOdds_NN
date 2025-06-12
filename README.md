# NFL game by game predictions

## Backgound
Vegas odds do a remarkable job predicting NFL games. Simply backing the money line favorite would have won 71.3% of the regular-season matchups in the 2024 season. That figure shows how the market pulls together information like fan sentiment, injury updates, and team reputation to set accurate odds. However, since the value considers population level betting trends, a metric that is not entirely based on subsintative team performances, there is likely significant room for improving that prediction. Simply put, sportsbooks aren't meerly trying to predict the victor but also balance the amount of money wagered on both  sides of a bet to insure proffits.

Here, I attempt to outperform the 71.3% baseline by not only considering Vegas odds, but additional features that aim to track team performances throughout the season. I've split the repository based on model type: 1) A Neural network, 2) a logistic regression, and 3) a decision tree. These approaches each have their strengths and weeknesses, but ultimately their output will be formated in a binary prediction and their perfromance will be measured to the baseline set by the vegas moneyline

## Data
Each model is trained and validated on the regular season games from 2020-2023. The test is performed against the 2024 season
