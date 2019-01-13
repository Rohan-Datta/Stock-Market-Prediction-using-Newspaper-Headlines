# Stock-Market-Prediction-using-Newspaper-Headlines
A project to create a predictive model for Dow Jones Industrial Average using daily headlines.

# Rationale
Dow Jones Industrial Average is a stock market index that collects the value of a list of 30 large and public companies based in the US. In this way it gives an idea of the trend that is going through the stock market.

News and global events, political or otherwise, play a major role in changing stock values. Every stock exchange is, after all, reflects how much trust investors are ready to put in other companies.

# Data and Pre-processing
The dataset used is [this](https://www.kaggle.com/aaron7sun/stocknews). It contains news headlines scraped from /r/worldnews and DJIA data downloaded from Yahoo Finance.

The data had some HTML tags that needed to be weeded out. Basic text cleaning was achieved through regex. Porter Stemmer was used to good effect.

Data points till 01-01-2015 were accumulated in the training subset and the rest were put aside for testing.

# Model
Count Vectorizer was used with bigrams and the best 235 features were picked to train the model. These values were mined after extensive hyperparameter tuning.

A simple Logistic Regression model was used with SAG (Stochastic Average Gradient) solver and inverse regularization parameter C = 0.31. These, too, were set after parameter tuning.

# Results
Training accuracy = 1030/1611 = 63.93%

Test accuracy = 224/378 = 59.25%

ROC AUC score = 0.575

# Review
There is not a huge correlation between the independent and dependent variables. This leads to an upper cap on accuracy.
This might also be the reason that Logistic Regression worked better than XGBoost, Random Forest, ANNs. Low correlation data prefers simpler models. This way, overfitting can be controlled.

Also, strangely, stopword removal actually worsened the performance of the model. This might be due to the smaller size of the data, and again, the excessive amounts of noise in it.

# References
* https://towardsdatascience.com/l1-and-l2-regularization-methods-ce25e7fc831c
* https://medium.com/deep-math-machine-learning-ai/chapter-9-1-nlp-word-vectors-d51bff9628c1
* https://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html
