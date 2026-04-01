# Feature Engineering for Time Series Forecasting in Python

Feature engineering is the process of creating additional input features
from raw time series data to improve the performance of predictive...

### Feature Engineering for Time Series Forecasting in Python
Feature engineering is the process of creating additional input features
from raw time series data to improve the performance of predictive
models.


<figcaption>Photo by <a
class="markup--anchor markup--figure-anchor"
rel="photo-creator noopener" target="_blank">Massimiliano Latella</a>
on <a
class="markup--anchor markup--figure-anchor"


Unlike static datasets, time series data has unique temporal
properties --- patterns like trends, seasonality, and lag
relationships --- that can be extracted and transformed into valuable
features. This article applies feature engineering techniques to
examples of time series including scaling, differencing, derivatives,
and memory embedding.

#### Why Feature Engineering Matters for Time Series
Time series forecasting is not just about feeding raw data into a model.
Transforming the data to highlight important patterns or trends can:

1\. Improve Model Accuracy by adding relevant features.

2\. Reveal Hidden Patterns like rates of change or lagged effects.

3\. Help Models Learn Temporal Dependencies, especially when using
non-sequential models like regression.

Good feature engineering ensures models are equipped to learn both
short-term and long-term relationships in the data.


#### Scaling Values
Time series values often vary in magnitude, and unscaled data can hinder
the performance of models, especially those that rely on gradient
descent (e.g., neural networks). Scaling methods include:

Min-Max Scaling: Rescales values to a specific range, e.g., \[0, 1\].

Standardization (Z-score): Rescales values to have a mean of 0 and
standard deviation of 1.

Python Example for Scaling:


When to Use:

Use Min-Max Scaling for data where the range matters.

Use Standardization when the scale is unknown or when working with
models sensitive to variance.

#### Looking at Changes in Values
Instead of analyzing absolute values, focusing on changes can remove
trends and reveal stationarity. Differencing calculates the difference
between consecutive values:


Python Example for Differencing:


First-order differencing removes trends to make the data stationary and
highlighting short-term changes in the series. This transformation can
be reveal patterns that might be obscured in the original series.
However, differencing inherently reduces the length of the dataset, as
the first observation cannot be differenced. Additionally, while it's
effective for revealing short-term fluctuations, excessive differencing
may lead to a loss of valuable long-term information contained in the
original series. Therefore, analysts must carefully balance the
advantages of improved stationarity and short-term insight against the
potential loss of data points and long-term trends when applying this
technique.

#### Derivatives: Rate of Change and Acceleration
Derivatives measure the rate of change in a time series, which can
highlight momentum or acceleration patterns.

First Derivative: Measures the rate of change.

Second Derivative: Measures the rate of change of the rate of change
(acceleration).

Python Example for Derivatives:


First derivatives help capture trends and momentum.

Second derivatives detect points of inflection or changes in
acceleration.

#### Embedding Prior Values: Building "Memory"
Embedding previous observations as features allows models to "remember"
past values. This is especially important for models that do not
inherently capture temporal dependencies (e.g., regression).

Embedding Lagged Values: Create additional features for previous time
steps:


Python Example for Lagged Features:


Lagged values embed past information into the current observation so we
can use regression and ML-based models. But adding too many lagged
features will cause overfitting.

#### Other Common Feature Engineering Techniques
Rolling Statistics: Calculate rolling means, variances, or medians over
a window to smooth the series.


Extracting Seasonality: Decompose a series into trend, seasonal, and
residual components.


Fourier Transforms: Use Fourier transformations to identify dominant
frequencies in seasonal data.

Time-Based Features: Extract calendar-related features like month, day
of the week, or hour to capture seasonality.


Lagged Differences --- Combining lags with differencing can capture both
memory and change.

The best features depend on the nature of the data and the modeling
task.

#### Real World Example: ERCOT Electric Load data
Let's apply this to a real dataset of data from the electric grid in
Texas.


Using this data, we can apply the transforms we use in this article.



Now we can create a simple forecast using Exponential Smoothing.



#### Next Steps
Feature engineering transforms raw time series data into meaningful
representations, enabling models to learn richer temporal patterns.
Techniques like scaling, differencing, derivatives, and memory embedding
improve predictive accuracy.

Pandas makes these transformations straightforward for time series
datasets.

#### Beehive Example (continued)
You create some additional features to help you with your analysis.

- Lagged features: Yesterday's weight affects today.
- Rate of Change: First derivative of weight to detect honey production
  speed.


#### Related Posts
This article is part of a series of posts on time series forecasting.
Here is the list of articles in the order they were designed to be read.

1.  [[Time Series for Business Analytics with
    Python](https://medium.com/@kylejones_47003/time-series-for-business-analytics-with-python-a92b30eecf62?source=your_stories_page-------------------------------------)]
2.  [[Time Series Visualization for Business Analysis with
    Python](https://medium.com/@kylejones_47003/time-series-visualization-for-business-analysis-with-python-5df695543d4a?source=your_stories_page-------------------------------------)]
3.  [[Patterns in Time Series for
    Forecasting](https://medium.com/@kylejones_47003/patterns-in-time-series-for-forecasting-8a0d3ad3b7f5?source=your_stories_page-------------------------------------)]
4.  [[Imputing Missing Values in Time Series Data for Business Analytics
    with
    Python](https://medium.com/@kylejones_47003/imputing-missing-values-in-time-series-data-for-business-analytics-with-python-b30a1ef6aaa6?source=your_stories_page-------------------------------------)]
5.  [[Measuring Error in Time Series Forecasting with
    Python](https://medium.com/@kylejones_47003/measuring-error-in-time-series-forecasting-with-python-18d743a535fd?source=your_stories_page-------------------------------------)]
6.  [[Univariate and Multivariate Time Series Analysis with
    Python](https://medium.com/@kylejones_47003/univariate-and-multivariate-time-series-analysis-with-python-b22c6ec8f133?source=your_stories_page-------------------------------------)]
7.  [[Feature Engineering for Time Series Forecasting in
    Python](https://medium.com/@kylejones_47003/feature-engineering-for-time-series-forecasting-in-python-7c469f69e260?source=your_stories_page-------------------------------------)]
8.  [[Anomaly Detection in Time Series Data with
    Python](https://medium.com/@kylejones_47003/anomaly-detection-in-time-series-data-with-python-5a15089636db?source=your_stories_page-------------------------------------)]
9.  [[Dickey-Fuller Test for Stationarity in Time Series with
    Python](https://medium.com/@kylejones_47003/dickey-fuller-test-for-stationarity-in-time-series-with-python-4e4bf1953eed?source=your_stories_page-------------------------------------)]
10. [[Using Classification Model for Time Series Forecasting with
    Python](https://medium.com/@kylejones_47003/using-classification-model-for-time-series-forecasting-with-python-d74a1021a5c4?source=your_stories_page-------------------------------------)]
11. [[Measuring Error in Time Series Forecasting with
    Python](https://medium.com/@kylejones_47003/measuring-error-in-time-series-forecasting-with-python-18d743a535fd?source=your_stories_page-------------------------------------)]
12. [[Physics-informed anomaly detection in a wind turbine using Python
    with an autoencoder
    transformer](https://medium.com/@kylejones_47003/physics-informed-anomaly-detection-in-a-wind-turbine-using-python-with-an-autoencoder-transformer-06eb68aeb0e8?source=your_stories_page-------------------------------------)]

#### Example with synthetic data

### Thank you for being a part of the community
*Before you go:*

- Be sure to **clap** and **follow** the writer ️👏**️️**
- [Follow us: [**X**](https://x.com/inPlainEngHQ) \|
  [**LinkedIn**](https://www.linkedin.com/company/inplainenglish/) \|
  [**YouTube**](https://www.youtube.com/channel/UCtipWUghju290NWcn8jhyAw) \|
  [**Newsletter**](https://newsletter.plainenglish.io/) \|
  [**Podcast**](https://open.spotify.com/show/7qxylRWKhvZwMz2WuEoua0)]
- [[**Check out CoFeed, the smart way to stay up-to-date with the latest
  in tech**](https://cofeed.app/)
  **🧪**]
- [[**Start your own free AI-powered blog on
  Differ**](https://differ.blog/)
  🚀]
- [[**Join our content creators community on
  Discord**](https://discord.gg/in-plain-english-709094664682340443) 🧑🏻‍💻]
- [For more content, visit
  [**plainenglish.io**](https://plainenglish.io/) +
  [**stackademic.com**](https://stackademic.com/)]
