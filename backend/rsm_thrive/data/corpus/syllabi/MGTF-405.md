---
id: "MGTF 405"
code: "MGTF 405"
title: "Business Forecasting"
department: "MGTF"
programme: "Rady MQF (Master of Quantitative Finance)"
units: 4
schedulable: true
is_core: false
technical_level: 4
workload: "heavy"
workload_basis: "Quantitative MFin coursework; letter grades only."
grading: "See department syllabus."
offerings: [{"term": "FA26", "season": "FA", "instructor": "Timmermann, Allan"}]
prerequisites: "Restricted to MFin/MBA or by consent of instructor."
topics: ["time-series analysis", "forecasting models", "forecast evaluation", "financial data"]
skills: ["building and evaluating forecasting models for finance"]
tools: ["R", "Python"]
career_tags: ["finance", "machine-learning"]
notes: "Pre-approved MSBA elective (MFin course — enrollment by consent). Scheduled for Fall 2026; no Winter or Spring section is on the schedule."
syllabi: ["MGTF 405 Business Forecasting (Timmermann) FA26.txt"]
---
# MGTF 405 — Business Forecasting

**4 units** · Rady MQF (Master of Quantitative Finance) · offered FA26, varies, varies, varies

## What this course covers

Master of Finance course on forecasting for business and financial decisions: time-
series methods, forecast evaluation, and applications to financial and economic data.

## What you should be able to do afterwards

- building and evaluating forecasting models for finance

## Topics covered

time-series analysis, forecasting models, forecast evaluation, financial data.

## Tools used

R, Python.

## Prerequisites

Restricted to MFin/MBA or by consent of instructor.

## How it is graded

See department syllabus.

## Workload

Heavy. Quantitative MFin coursework; letter grades only.

## Offerings

- **FA26** — Timmermann, Allan
- **varies**
- **varies**
- **varies**

## Syllabus text

_Verbatim from the syllabus PDF supplied by the Rady graduate programmes office._

### MGTF 405 Business Forecasting (Timmermann) FA26

MGTF 405 BUSINESS FORECASTING

Fall 2026, Master of Quantitative Finance

PROFESSOR ALLAN TIMMERMANN
EMAIL: atimmermann@ucsd.edu
TEACHING ASSISTANT: Luka Vulicevic | EMAIL: lvulicevic@ucsd.edu

COURSE OBJECTIVES
Forecasts of uncertain future outcomes or events form a key input into most business and investment decisions and
so affect all areas of finance and business practice. Stock pickers and investment analysts need to forecast future
returns on the stocks they consider investing in. Airlines need to forecast fuel prices to determine whether to hedge
their exposure to future oil prices. Companies need to forecast both the expected value and volatility of their future
cash flows to determine whether to use debt or equity in their financing decisions. The Federal Reserve decides
current interest rates based on their predictions of future inflation and economic activity. Home builders need to plan
their construction activities years in advance and so have to predict both house prices and demand for homes before
committing millions of dollars into essentially irreversible investment projects. Pharmaceutical and IT firms are
faced with both technological uncertainty (which products are feasible?) and uncertainties about future market size,
competitors' behavior, and market shares. Predictions of these variables will affect which projects should be
launched immediately, which ones should be canceled, and which ones should be postponed.
This course introduces students to quantitative methods for producing their own state-of-the-art forecasts based on
information from sources such as past values of the variable(s) being predicted, surveys, market information and
other relevant economic data. It also teaches course participants to become critical consumers of forecasts reported
in the media and by professional forecasters. Are such forecasts of much value and, if so, how good is a forecaster's
track record?
The course relies extensively on quantitative methods. Empirical methods such as time-series models, machine
learning, regression analysis, nonlinear estimation and graphical tools lie at the core of the course and will be used
extensively. Reflecting recent developments in the field, the course also introduces prediction markets and large
language models (LLMs) as emerging tools and sources of information for forecasters — both as objects of study in
their own right and as inputs that increasingly compete with, and complement, traditional statistical forecasts.

COURSE MATERIALS
Elliott, Graham, and Allan Timmermann, Economic Forecasting. Princeton University Press, 2016.
This book (referred to as ET below) provides a comprehensive coverage of a wide range of quantitative forecasting
topics. The coverage sometimes becomes quite technical and on occasion we will use the book for background
reading. A set of background papers can be downloaded from the web.

COURSE GRADES
●
●
●
●

Class participation (10%)
Assignments 1-3 (25%)
Midterm exam (25%)
Forecasting project (40%)

TEACHING ASSISTANT
The teaching assistant for this course is Luka Vulicevic. Luka will be available to answer questions in person and
via e-mail during the week. His e-mail is lvulicevic@ucsd.edu .

PROFESSOR CONTACT INFORMATION
E-mail: atimmermann@ucsd.edu (Office: Otterson Hall 4S144)

STUDENTS WITH DISABILITIES
A student who has a disability or special need and requires an accommodation in order to have equal access to the
classroom must register with the Office for Students with Disabilities (OSD). The OSD will determine what
accommodations may be made and provide the necessary documentation to present to the faculty member.
The student must present the OSD letter of certification and OSD accommodation recommendation to the
appropriate faculty member in order to initiate the request for accommodation in classes, examinations, or other
academic program activities. No accommodations can be implemented retroactively.
Please visit the OSD Web site (osd.ucsd.edu) for further information or contact the Office for Students with
Disabilities at (858) 534-4382 or fosorio@ucsd.edu.

ACADEMIC INTEGRITY
Integrity of scholarship is essential for an academic community. As members of the Rady School, we pledge
ourselves to uphold the highest ethical standards. The University expects that both faculty and students will honor
this principle and in so doing protect the validity of University intellectual work. For students, this means that all
academic work will be done by the individual to whom it is assigned, without unauthorized aid of any kind.
The complete UCSD Policy on Integrity of Scholarship can be viewed at: http://wwwsenate.ucsd.edu/manual/appendices/app2.htm#AP14

MGTF 405 COURSE OUTLINE
Week 1, September 28: Introducing the Forecasting Problem
●

Course overview
○ Challenges facing forecasters
○ The forecasting process
○ Costs of making forecast errors: The loss function
○ Forecast horizon
○ Formal forecasting models
○ Desirable Properties of Forecasts
● The changing forecasting landscape: model-based forecasts, judgment, prediction markets, and LLMs as
competing and complementary sources of predictions
● Simple Measures of Forecast Performance: MSE, MAE, directional accuracy
○ What does a good forecast look like? Graphical Analysis Tools
● Cold-start forecasting problems
● In-sample versus out-of-sample forecasts (backtesting)
Readings:
ET chapter 2, 15.1, 15.3.1, 16.1, 16.2.
Background readings
A. Timmermann, 2018, Forecasting Methods in Finance. Annual Review of Financial Economics, 10, 449-479.
Breitung, C., 2026, Text is all you need: Asset pricing without Returns. SSRN Working Paper.

Hong, H., J. D. Kubik, and A. Solomon, 2000, “Security Analysts’ Career Concerns and Herding of Earnings
Forecasts.” Rand Journal of Economics, 31, 121-144.

Week 2, October 5: Time-series Forecasting Models: Learning from Past
Information
●

Time-series methods in forecasting
○ Mean, variance, autocorrelations, persistence
○ Covariance stationarity
○ White noise: the building block
○ Wold representation theorem
○ Moving Average and Autoregressive models
○ Chain rule of forecasting
○ Random Walk model with and without trend
○ Unit root tests
○ Seasonality and Trend
Readings:
ET chapter 7.1 - 7.4.

Week 3, October 12: Selecting a good Forecasting Model
●

Linear Regression Model Used in Forecasting
○ OLS, Maximum Likelihood and GMM estimation
○ Data Transformation
● Model selection
○ Stepwise procedures, Information criteria, LASSO, Elastic Net, Bagging
● Machine Learning methods
Assignment 1 due
Readings:
ET Chapter 6.
Background readings
Goyal, A. and I. Welch, 2008, A Comprehensive Look at the Empirical Performance of Equity Premium
Prediction. Review of Financial Studies 21(4), 1455-1508.
Kalfi, S. Y., A. Timmermann, and T van der Zwan, 2026, Overhyped? Can ML Models Reliably Predict Stock
Returns? Working paper, UCSD.

Week 4, October 19: Unobserved variables, Machine Learning, Nonlinear
Forecasting Models
● Machine learning and nonlinear forecasting models (cont.)
● Foundation Models — Time-series foundation models (TSLMs, e.g. TimeGPT-style architectures)
● Kalman filter – unobserved components models
● Markov switching models
● Exponential Smoothing
Readings:
ET chapter 7.5, 8.3, 8.5, Appendix A1 and A2 on the Kalman filter

Background readings
Ang, A. and A. Timmermann, 2012, Regime Changes and Financial Markets. Annual Review of Financial
Economics 4: 313-337.
Aruoba, S.B, F.X. Diebold, and C. Scotti, 2009, Real-time measurement of business conditions. Journal of
Business and Economic Statistics 27, 417-427.
Gu, S., B. Kelly, and D. Xiu, 2020, Empirical Asset Pricing via Machine Learning. Review of Financial Studies
33, 2223-2273.
Rahimikia, E., Ni, Hao, and Weiguan Wang, 2026, Re(Visiting) Time Series Foundation Models in Finance.
Working paper, University College London.
Carriero, A., D. Pettenuzzo, and S. Shekhar, 2025, Macroeconomic Forecasting with Large Language Models.
arXiv:2407.00890.

Week 5, October 26: Multivariate forecasting models
● Vector Auto Regressions (VARs)
2-hour midterm exam.
Readings:
ET chapter 9.1, 9.2.

Week 6, November 2: Vector Autoregressions and Factor models (cont.)
●

Vector Auto Regressions (VARs)
○ Classical vs. Bayesian methods
○ Cointegration and error correction models
○ Spurious correlation
● Forecasting with very large data sets
○ Factor models
○ Panel data
○ Real-time data
● Spatial forecasting
Assignment 2 due
Readings:
ET chapter 9.3, 9.6, 10.
Background readings
Ghezzi, F., A. Timmermann, and M. Yang, 2026, The Hourly Economy: Measuring Local Economic Activity at
High Frequency. UCSD working paper.
Karlsson, S., 2013, Forecasting with Bayesian Vector Autoregressions. Chapter 15 (pages 791-898) in G. Elliott
and A. Timmermann (eds.): Handbook of Economic Forecasting, vol. 2.
Stock, J.H., and M.W. Watson, 2006, Forecasting with Many Predictors. Chapter 10 (pages 515-554) in G.
Elliott, C.W.J. Granger and A. Timmermann (eds.): Handbook of Economic Forecasting, vol. 1.

Week 7, November 9: Prediction Markets, LLM-Based Forecasting, and
Event/Density/Volatility Forecasting
●

Event Forecasting models
○ Linear probability model
○ Logit/probit

○ Sign predictions
○ Brier Score
● Prediction Markets
○ Growth of Kalshi/Polymarket-style markets into macro, election, and business-outcome contracts;
● LLMs as forecasters: judgmental/event forecasting vs. superforecasters (Metaculus, ForecastBench
tournaments) (Karger et al, 2025)
● LLMs as measurement tools: text-to-signal econometrics, training-data leakage, and validation-sample
requirements for valid inference (Ludwig et al, 2025)
● Interval and quantile forecasts
● Density and volatility forecasting
Readings:
ET chapter 13
Bartlett, R. and M. O’Hara, 2025, Adverse selection in prediction markets: Evidence from Kalshi. Unpublished
working paper, Stanford and Cornell.
Diercks, A.M., J. D. Katz, and J. W Wright, 2026, Kalshi and the Rise of Macro Markets. Federal Reserve
Board, Washington DC.
Gomez Cram, R., Y. Guo, T. I. Jensen, and H. Kung, 2025, Financial Prediction Markets: A new Measure of
Earnings Expectations. SSRN paper.
New background readings on LLM-based forecasting:
Ludwig, J., S. Mullainathan, and A. Rambachan, 2025, Large Language Models: An Applied Econometric
Framework. Forthcoming, Annual Review of Economics, 2026.
Karger, E., H. Bastani, C. Yueh-Han, Z. Jacobs, F. Zhang, and P.E. Tetlock, 2025, ForecastBench: A Dynamic
Benchmark of AI Forecasting Capabilities. arXiv:2409.19839.
Forecasting Research Institute / Good Judgment, 2025, Human vs. AI Forecasts (comparison of superforecaster
and frontier-LLM difficulty-adjusted Brier scores).
Jadhav, A. and V. Mirza, 2025, Large Language Models in Equity Markets: Applications, Techniques, and
Insights. Frontiers in Artificial Intelligence (survey of 84 studies, 2022–early 2025)

Week 8, November 16: Forecast Combination
●
●
●
●
●

Forecast combination: A simple method that often beats the single best forecaster
Why combine forecasts?
Forecast encompassing – does one forecast dominate?
Portfolios of forecasts
Regression methods for forecast combination
○ Combining human, model, market, and LLM forecasts (“wisdom of the silicon crowd”)
Assignment 3 due
Readings:
ET chapter 14.
Background readings
Diebold, F.X. and M. Shin, 2019, Machine learning for regularized survey forecast combination: Partiallyegalitarian LASSO and its derivatives. International Journal of Forecasting 1679-1691.
Elliott, G., A. Gargano and A. Timmermann, 2013, Complete Subset Regressions. Journal of Econometrics 177,
357-373.

Makridakis S., E. Spiliotis, and V. Assimakopoulos, 2018, The M4 Competition: Results, findings, Conclusion,
and Way Forward. International Journal of Forecasting 802-808.
Rapach, D., J.K. Strauss, and G. Zhou, 2010, Out-of-Sample Equity Premium Prediction: Combination
Forecasts and Links to the Real Economy. Review of Financial Studies, 23(2), 821-862.

Week 9, November 23: Forecast evaluation and forecast comparison
● Forecast Evaluation
● Properties of optimal forecasts
● Regression tests
● Measures of forecast precision
● Comparing predictive accuracy
● Encompassing tests
● Look-ahead bias
Readings:
ET chapter 15.3, 17.1 – 17.4

Week 10, November 30: Model Breakdown. Pitfalls in forecasting. Data mining.
●
●
●
●
●
●
●
●
●

Breaks in forecasting models
Detection
Importance for forecasting
Skill or Luck? The Effect of Data Mining on Forecasting
Data mining defined
Newsletter, lucky penny examples
In-sample versus out-of-sample forecasts (recap)
Cross-validation
Methods for handling data mining
○ Memorization and training-data leakage as a data-mining-like pitfall specific to LLM-based
forecasts
Readings:
ET chapter 17.5-17.10, 19
Background readings
McLean, R.D., and J. Pontiff, 2016, Does Academic Research Destroy Stock Return Predictability? Journal of
Finance 71, 5-32.

Week 11: Final Project is Due on Monday, December 7 at 8 pm.

