# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# %%
#QUESTION 01 SET
#1.Compute the average prices and scores by Neighborhood;
# which borough is the most expensive on average? 
# Create a kernel density plot of price and log price, grouping by Neighborhood.

# --- Load dataset ---
# Read in the CSV file (make sure it's in the same folder)
df = pd.read_csv("NYC_Airbnb.csv")

# Clean column names 
df = df.rename(columns={"Neighbourhood ": "Neighborhood"})

# Check columns to confirm fix
print(df.columns)

# %%
# --- Compute averages by Neighborhood ---
# Group by borough and compute mean Price and Review Scores
grouped = df.groupby("Neighborhood")[["Price", "Review Scores Rating"]].mean()
print(grouped)
#ANSWER:
# Manhattan is the most expensive borough on average (~183.66).
# Brooklyn and Staten Island are also relatively expensive,
# while the Bronx is the cheapest. Review scores are similar across boroughs, all around the low 90s.


# %%
# --- KDE plot for Price ---
sns.kdeplot(data=df, x="Price", hue="Neighborhood")  # distribution of price
plt.show()

# %%
# --- Log transform ---
df["log_price"] = np.log(df["Price"] + 1)  # reduce skew

# %%
# --- KDE plot for Log Price ---
sns.kdeplot(data=df, x="log_price", hue="Neighborhood")
plt.show()

# %%
# Q2: Regress price on Neighborhood  by creating the appropriate dummy/one-hot-encoded variables,
# without an intercept in the linear model. 
# Compare the coefficients in the regression to the table from part 1. 
# What pattern do you see? 
# What are the coefficients in a regression of a continuous variable on one categorical variable?

# basically want to see if the regression gives the same averages computed before

# --- Create dummy variables for Neighborhood ---
# regression can't use text so we must Convert Brox, Brooklyn, Manhattan, Queens, Staten Island into dummy variables
# dummy variables are 0/1 indicators for each category

# drop_first=False keeps all columns, so we get a coefficient for each neighborhood instead of using one as a baseline
X = pd.get_dummies(df["Neighborhood"], drop_first=False)

#selects the thing we are predicting 
# X = inputs (Neighborhood); y = output (price)
y = df["Price"]

# %%
# Fit linear regression model 
# fit_intercept=False means no baseline, no constant term; each coefficient will represent the average price for that neighborhood directly
model = LinearRegression(fit_intercept=False)

#this is where the model learns; looks at X (which neighborhood each row is) and y (price)
#and figures out what number should each neighborhood have
model.fit(X,y)
# %%
print(model.coef_)
#[ 75.27, 127.75, 183.66, 96.86, 146.17 ] 
# These coefficients represent the average price for each neighborhood

# %%
# shows which coefficient corresponds to which neighborhood
print(X.columns)
#Index(['Bronx', 'Brooklyn', 'Manhattan', 'Queens', 'Staten Island']

#ANSWER: 
# The coefficients correspond to the Neighborhood columns in the same order. Each coefficient represents the average price for that neighborhood.
# These values match the averages from part 1, showing that when we regress a continuous variable on a categorical variable using dummy variables without an intercept, the coefficients are equal to the group means.

# %%
#Q3: Repeat part 2, but leave an intercept in the linear model. 
# How do you have to handle the creation of the dummies differently? 
# What is the intercept? Interpret the coefficients. 
# How can I get the coefficients in part 2 from these new coefficients?

# with intercept, we need to drop one dummy variable to avoid the dummy variable trap.

#drop_first=True means we drop the first category (Bronx) and use it as the baseline.
X = pd.get_dummies(df["Neighborhood"], drop_first=True)

#selects the thing we are predicting 
# X = inputs (Neighborhood); y = output (price)
y = df["Price"]

# %%
# Fit linear regression model with intercept

model = LinearRegression() #intercept is included by default
model.fit(X,y)

#Results
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_) 
print("Columns:", X.columns)

#The intercept = 75.27, which is the average price for the baseline category (Bronx).
# The coefficients represent the difference in average price compared to the baseline (Bronx):
# So the intercept = the average price for the Bronx, and each coefficient tells us how much more expensive that neighborhood is compared to the Bronx.

#ANSWER:
# When including an intercept, we must drop one dummy variable to avoid multicollinearity. In this case, Bronx is the baseline category since it was dropped.
#The intercept represents the average price in the Bronx, which is about 75.27. Each coefficient represents the difference in price between that neighborhood and the Bronx. For example, Manhattan is about 108.39 dollars more expensive than the Bronx on average.
#To recover the coefficients from part 2, we add each coefficient to the intercept, which gives the average price for each neighborhood.

# %%
#Q4: Split the sample 80/20 into a training and a test set. Run a regression of Price on Review Scores Rating and Neighborhood . 
# What is the R^2 and RMSE on the test set? 
# What is the coefficient on Review Scores Rating? 
# What is the most expensive kind of property you can rent?

#asked to: split data --> train/test
#regress price on: review scores rating (numeric) and neighborhood (categorical --> dummies)
#evaluate on test set: R^2, RMSE
#Interpret coefficient on review scores rating
#Find most expensive kind of property 

#create dummy variables for Neighborhood, drop one for baseline
X = pd.get_dummies(df[["Review Scores Rating", "Neighborhood"]], drop_first=True)

#target variable
y = df["Price"]

# %%
# split data into training and test sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Fit Model
model = LinearRegression()
model.fit(X_train, y_train)

# %%
#Predictions on test set
y_pred = model.predict(X_test)

#Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("R^2 on test set:", r2)
print("RMSE on test set:", rmse)

# %%
# Coefficients 
coef_df = pd.DataFrame({
    "Variable": X.columns,
    "Coefficient": model.coef_
})
print(coef_df)

print("Intercept:", model.intercept_)

#ANSWER:
#The R² on the test set is about 0.046, and the RMSE is about 140.92, indicating that the model does not explain much of the variation in price.
# The coefficient on Review Scores Rating is about 1.21, meaning that for each one-point increase in rating, the price increases by about $1.21, holding neighborhood constant.
#The most expensive neighborhood is Manhattan, since it has the largest positive coefficient (about 107.41) relative to the baseline (Bronx).
#Compared to the previous model, the neighborhood coefficients are smaller, suggesting that some of the differences in price across neighborhoods are explained by review scores.

# %%
#Q5: Run a regression of Price on Review Scores Rating and Neighborhood  and Property Type. 
# What is the R^2 and RMSE on the test set? 
# What is the coefficient on Review Scores Rating? 
# What is the most expensive kind of property you can rent?

#now we make the model include property type as well, which is another categorical variable that we need to convert to dummies

# create dummy variables for Neighborhood and Property Type, drop one for baseline
X = pd.get_dummies(df[["Review Scores Rating", "Neighborhood", "Property Type"]], drop_first=True)

#create target variable
y = df["Price"]


# %%
# split data into training and test sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Fit in the model
model = LinearRegression()
model.fit(X_train, y_train)
# %%
#Predictions on test set
y_pred = model.predict(X_test)
# %%
#Evaluate model performance

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("R^2 on test set:", r2)
print("RMSE on test set:", rmse)    

#R^2 on test set: 0.054242713551245325
#RMSE on test set: 140.30266238276283

# %%
# Coefficients
coef_df = pd.DataFrame({
    "Variable": X.columns,
    "Coefficient": model.coef_
})
print(coef_df)      

print("Intercept:", model.intercept_)

#ANSWER:
#The R² on the test set is about 0.054, and the RMSE is about 140.30, indicating that the model explains very little of the variation in price.

#The coefficient on Review Scores Rating is about 1.20, meaning that for each one-point increase in rating, the price increases by about $1.20, holding neighborhood and property type constant.

#The most expensive property type is Bungalow, since it has the largest positive coefficient (about 83.97) relative to the baseline category.

# %%
#Q6: What does the coefficient on Review Scores Rating mean if it changes from part 4 to 5?
#  Hint: Think about how multiple linear regression works.

#asking why did the coefficient on review scores rating change when you added property type
#idea: in multiple linear regressions -each coefficient is interpreted holding all other variables constant 
# what changed from 4 to 5?
# 4: controlled for neighborhood only
# 5: controlled for neighborhood and property type
# so now the model separates: effects of ratings and effects of property type 

#ANSWER:
# The coefficient on Review Scores Rating changes from part 4 to part 5 because the model now includes an additional variable, Property Type. 
# In multiple linear regression, each coefficient represents the effect of that variable while holding all other variables constant.
# In part 4, the coefficient on review scores may have captured some of the effect of property type. 
# In part 5, after adding Property Type to the model, the coefficient on review scores reflects its effect after accounting for both neighborhood and property type. 
# This change shows that some of the variation previously attributed to review scores is actually explained by property type.

# %%
#Q7:We've included Neighborhood  and Property Type separately in the model. 
# How do you interact them, so you can have "A bedroom in Queens" or "A townhouse in Manhattan". 
# Split the sample 80/20 into a training and a test set and run a regression including that kind of "property type X neighborhood" dummy, plus Review Scores Rating. 
# How does the slope coefficient for Review Scores Rating, the R^2and the RMSE change? 
# Do they increase significantly compares to part 5? 
# Are the coefficients in this regression just the sum of the coefficients for Neighbourhood  and Property Type from 5? 
# What is the most expensive kind of property you can rent?

# instead of treating: neighborhood, property type
# seperately commbinations like Bungalow in Manhattan, Townhouse in Queens, etc.

#create inetraction variable 
#combine neighborhood and property type into one column
df["Interaction"] = df["Neighborhood"] + "_" + df["Property Type"]

#create dummies and drop one for baseline
X = pd.get_dummies(df[["Review Scores Rating", "Interaction"]], drop_first=True)

# target variable
y = df["Price"]
# %%
# --- Train/test split ---
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)

# --- Fit model ---
model = LinearRegression()
model.fit(X_train, y_train)

# --- Predictions ---
y_pred = model.predict(X_test)
# %%
# --- Evaluation ---
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("R^2:", r2)
print("RMSE:", rmse)

# %%
# --- Coefficients ---
coef_df = pd.DataFrame({
    "Variable": X.columns,
    "Coefficient": model.coef_
})
print(coef_df)

print("Intercept:", model.intercept_)

#ANSWER:
#Including interaction terms allows the model to capture specific combinations of neighborhood and property type, such as a loft in Manhattan or a bungalow in Queens.
#The R² increases slightly to about 0.0546 and the RMSE decreases slightly to about 140.28 compared to part 5, indicating only a very small improvement in model performance.
#The coefficient on Review Scores Rating is about 1.20, which is very similar to the previous model, suggesting that ratings still have a small effect on price.
#The coefficients in this model are not simply the sum of the coefficients for Neighborhood and Property Type from part 5. Instead, each coefficient represents the effect of a specific combination of neighborhood and property type.
#The most expensive property is a loft in Manhattan, since it has the largest coefficient (about 285.32).

# %%

#QUESTION 2 SET

#Q1: Load cars_hw.csv. These data were really dirty, and I've already cleaned them a significant amount in terms of missing values and other issues, but some issues remain (e.g. outliers, badly skewed variables that require a log or arcsinh transformation) 
# Note this is different than normalizing: there is a text below that explains further. 
# Clean the data however you think is most appropriate

# --- Load dataset ---
df = pd.read_csv("cars_hw.csv")

# --- Check data ---
print(df.head())
print(df.describe())

# %%
# --- Drop unnecessary column ---
# "Unnamed: 0" is just an index column and does not contain useful information
df = df.drop(columns=["Unnamed: 0"])

# --- Fix No_of_Owners ---
# This column is stored as text (e.g., "1st", "2nd"), which cannot be used in regression
# We remove the text and keep only the numeric part, then convert to integers
df["No_of_Owners"] = df["No_of_Owners"].str.replace(r"\D", "", regex=True).astype(int)

# --- Handle skewed variables ---
# Price and Mileage_Run are right-skewed (many small values, few very large ones)
# Taking the log compresses large values and makes the distribution more symmetric
df["log_price"] = np.log(df["Price"] + 1)
df["log_mileage"] = np.log(df["Mileage_Run"] + 1)

# --- Remove extreme outliers ---
# Very large prices can distort the model, so we remove the top 1% of values
price_cutoff = df["Price"].quantile(0.99)
df = df[df["Price"] < price_cutoff]
# %%
# --- Check cleaned data ---
# This helps confirm that the data looks more reasonable after cleaning
print(df.describe())

#ANSWER:
#After cleaning the data, the summary statistics show that extreme values in Price have been reduced, as the maximum value decreased from about 2.94 million to about 1.98 million. This indicates that removing the top 1% of values successfully reduced the impact of outliers.

#The log transformations of Price and Mileage_Run result in more compressed distributions with smaller standard deviations, suggesting reduced skewness. This makes the variables more suitable for linear regression, as extreme values have less influence on the model.

#Overall, the cleaned data appears more balanced and appropriate for modeling.

# %%
#Q2:Summarize the Price variable and create a kernel density plot. 
# Use .groupby() and .describe() to summarize prices by brand (Make). 
# Make a grouped kernel density plot by Make. 
# Which car brands are the most expensive? What do prices look like in general?
#Split the data into an 80% training set and a 20% testing set.

# --- Summarize Price ---
# describe() gives count, mean, std, min, max, etc.
print(df["Price"].describe())


# %%
#  KDE plot for Price 
# Shows distribution (shape) of price
sns.kdeplot(df["Price"])
plt.title("Distribution of Car Prices")
plt.xlabel("Price")
plt.show()
# %%
#  Group by Make (brand) 
# groupby() groups data by brand
# describe() summarizes price within each brand
price_by_make = df.groupby("Make")["Price"].describe()
print(price_by_make)

# %%
#  KDE plot grouped by Make 
# Shows how price distributions differ by brand
sns.kdeplot(data=df, x="Price", hue="Make")
plt.title("Price Distribution by Car Brand")
plt.xlabel("Price")
plt.show()

#ANSWER: 
# The Price variable shows a right-skewed distribution, with most cars concentrated between about 400,000 and 900,000 and a long tail extending toward higher values. This indicates that while most cars are moderately priced, there are a few expensive cars that increase the average price.
# The kernel density plot confirms this pattern, showing a peak in the mid-price range and a gradual decline as prices increase.

# When grouping by Make, some brands have distributions shifted toward higher prices. Brands such as Jeep, MG Motors, and Skoda appear to be more expensive, as their distributions extend further to the right. In contrast, brands like Maruti Suzuki and Hyundai are more concentrated at lower price ranges.

# Overall, car prices are right-skewed, with most cars being moderately priced and a smaller number of high-priced vehicles.

# %%
#Q3: Split the data into an 80% training set and a 20% testing set.

#  Train/Test Split (80/20) 
from sklearn.model_selection import train_test_split

X = df.drop(columns=["Price"])  # predictors
y = df["Price"]                # target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# %%
#Q4:Make a model where you regress price on the numeric variables alone; what is the R^2 and RMSE on the training set and test set? 
# Make a second model where, for the categorical variables, you regress price on a model comprised of one-hot encoded regressors/features alone
#  (you can use pd.get_dummies(); be careful of the dummy variable trap); 
# what is the R^2 and RMSE on the test set? Which model performs better on the test set? 
# Make a third model that combines all the regressors from the previous two; 
# what is the R^2 and RMSE on the test set? 
# Does the joint model perform better or worse, and by home much?


# %%
#  Model 1: Numeric variables only 

# Select numeric features (these are already numbers)
X = df[["Make_Year", "Mileage_Run", "No_of_Owners", "Seating_Capacity"]]

# Target variable we want to predict
y = df["Price"]

# Split data into training (80%) and testing (20%)
# Training data is used to fit the model, testing data evaluates performance
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create linear regression model
model = LinearRegression()

# Fit the model (learn relationship between X and y)
model.fit(X_train, y_train)

# Make predictions on training and test data
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Compute performance metrics
# R^2 measures how much variation in price is explained by the model
# RMSE measures average prediction error in dollars
print("Numeric Model Train R^2:", r2_score(y_train, y_train_pred))
print("Numeric Model Test R^2:", r2_score(y_test, y_test_pred))

print("Numeric Model Train RMSE:", np.sqrt(mean_squared_error(y_train, y_train_pred)))
print("Numeric Model Test RMSE:", np.sqrt(mean_squared_error(y_test, y_test_pred)))
# %%
#  Model 2: Categorical variables only 

# Convert categorical variables into dummy (0/1) variables
# drop_first=True avoids multicollinearity (dummy variable trap)
X = pd.get_dummies(df[["Make", "Color", "Body_Type", "Fuel_Type", "Transmission_Type"]], drop_first=True)

# Target variable remains the same
y = df["Price"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test set
y_test_pred = model.predict(X_test)

# Evaluate performance
print("Categorical Model Test R^2:", r2_score(y_test, y_test_pred))
print("Categorical Model Test RMSE:", np.sqrt(mean_squared_error(y_test, y_test_pred)))
# %%
#  Model 3: Combined numeric + categorical variables 

# Combine numeric and categorical features into one dataset
X_num = df[["Make_Year", "Mileage_Run", "No_of_Owners", "Seating_Capacity"]]
X_cat = pd.get_dummies(df[["Make", "Color", "Body_Type", "Fuel_Type", "Transmission_Type"]], drop_first=True)

X = pd.concat([X_num, X_cat], axis=1)

# Target variable
y = df["Price"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test set
y_test_pred = model.predict(X_test)

# Evaluate performance
print("Combined Model Test R^2:", r2_score(y_test, y_test_pred))
print("Combined Model Test RMSE:", np.sqrt(mean_squared_error(y_test, y_test_pred)))
# %%
#ANSWER:
# The numeric-only model has a training R² of about 0.322 and a test R² of about 0.272. The RMSE is approximately 283,612 on the training set and 276,598 on the test set. This indicates that numeric variables alone explain a limited portion of the variation in car prices.

#The categorical-only model has a test R² of about 0.621 and an RMSE of about 199,510. This model performs substantially better than the numeric model, suggesting that categorical variables such as Make and Body_Type are strong predictors of price.

#The combined model has a test R² of about 0.779 and an RMSE of about 152,466. This model performs the best, as it incorporates both numeric and categorical variables.

#Overall, the combined model improves performance significantly compared to the individual models. The R² increases from 0.621 to 0.779 and the RMSE decreases from about 199,510 to 152,466, showing that combining both types of variables provides a much better fit.

# %%
#Q5: Use the PolynomialFeatures function from sklearn to expand the set of numerical variables you're using in the regression. 
# As you increase the degree of the expansion, how do the R^2and RMSE change? 
# At what point does R^2 go negative on the test set? 
# For your best model with expanded features, what is the R^2 and RMSE? 
# How does it compare to your best model from part 4?

#degree 1 → original variables
#degree 2 → squares + interactions
#degree 3 → more complex terms
#Then see how performance change


# %%
#  Polynomial regression with numeric variables only 

# Use the same numeric variables from the earlier model
X = df[["Make_Year", "Mileage_Run", "No_of_Owners", "Seating_Capacity"]]
y = df["Price"]

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Try several polynomial degrees
for degree in [1, 2, 3, 4, 5]:
    
    # PolynomialFeatures creates squared terms, cubic terms, and interactions
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    
    # Fit on training data and transform both train and test sets
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    # Fit linear regression model on expanded features
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    
    # Predict on the test set
    y_test_pred = model.predict(X_test_poly)
    
    # Compute test performance
    r2 = r2_score(y_test, y_test_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    print("Degree:", degree)
    print("Test R^2:", r2)
    print("Test RMSE:", rmse)
    print()

#ANSWER:
# As the degree of the polynomial expansion increases, the model becomes more flexible and can capture more complex relationships. From degree 1 to degree 2 and 4, the R² increases and the RMSE decreases, indicating improved performance.

# However, after a certain point, the model begins to overfit, as seen by fluctuations in R² and RMSE for higher degrees such as 3 and 5. In this case, the R² does not become negative for any of the tested degrees.

#The best-performing polynomial model occurs at degree 4, with a test R² of about 0.308 and an RMSE of about 269,735.

#Compared to the best model from part 4 (the combined model), which had a test R² of about 0.779 and RMSE of about 152,466, the polynomial model performs significantly worse. This suggests that categorical variables provide much more predictive power than simply adding nonlinear transformations of numeric variables.

# %%
#Q6: For your best model so far, determine the predicted values for the test data and plot them against the true values.
# Do the predicted values and true values roughly line up along the diagonal, or not? 
# Compute the residuals/errors for the test data and create a kernel density plot. 
# Do the residuals look roughly bell-shaped around zero?
# Evaluate the strengths and weaknesses of your model.

# Use best model (combined variables) 

# Recreate combined features
X_num = df[["Make_Year", "Mileage_Run", "No_of_Owners", "Seating_Capacity"]]
X_cat = pd.get_dummies(df[["Make", "Color", "Body_Type", "Fuel_Type", "Transmission_Type"]], drop_first=True)

X = pd.concat([X_num, X_cat], axis=1)
y = df["Price"]

# Train/test split (same as before)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Fit model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# %%
#  Plot predicted vs actual values 
# If model is good, points should lie near the diagonal line
plt.scatter(y_test, y_pred)
plt.xlabel("True Price")
plt.ylabel("Predicted Price")
plt.title("Predicted vs True Prices")
plt.show()

# %%
#  Compute residuals 
# residual = actual - predicted
residuals = y_test - y_pred

# %%
#  Plot residual distribution 
# Should be roughly centered around 0 if model is good
sns.kdeplot(residuals)
plt.title("Residual Distribution")
plt.xlabel("Residuals")
plt.show()

# ANSWER: 
# The predicted values and true values show a clear positive relationship, with many points roughly aligning along the diagonal. This indicates that the model captures the general pattern in car prices, although there is some spread, especially at higher price levels, suggesting less accuracy for more expensive cars.

#The residuals are centered around zero and appear approximately bell-shaped, indicating that the model’s errors are reasonably well distributed. However, there is a slight right tail, suggesting that the model occasionally underestimates higher-priced vehicles.

#A strength of the model is that it explains a large portion of the variation in price, as seen from the relatively high R² value. Additionally, the residuals being centered around zero suggests that the model is not systematically biased.

#A weakness of the model is that there is still noticeable variability in predictions, particularly for higher-priced cars, which indicates that some important factors affecting price may not be included in the model. Overall, the model performs well but could be improved by incorporating additional features or more complex relationships.

# %%
#QUESTION 3 SET

#Q1: Find a dataset on a topic you're interested in. Some easy options are data.gov, kaggle.com, and data.world.

# For this question, I selected a dataset on student performance factors. The dataset includes variables such as Hours_Studied, Attendance, Sleep_Hours, Motivation_Level, and other demographic and behavioral factors, with Exam_Score as the target variable.

#I chose this dataset because it contains both numeric and categorical variables, making it well-suited for applying linear regression techniques similar to those used in previous questions. It also allows for analyzing how different factors influence academic performance.

# Load dataset 
df = pd.read_csv("StudentPerformanceFactors.csv")

#  Check data 
print(df.head())
print(df.info())
# %%
#Q2: Clean the data and do some exploratory data analysis on key variables that interest you. 
# Pick a particular target/outcome variable and features/predictors.
# Clean data 
# Drop rows with missing values to keep things simple
df = df.dropna()

# %%
#  Convert categorical variables 
# Turn all text variables into dummy (0/1) variables
df = pd.get_dummies(df, drop_first=True)
# %%
# %%
#  Define target and predictors 
# Target variable (what we want to predict)
y = df["Exam_Score"]

# Predictor variables (everything else)
X = df.drop(columns=["Exam_Score"])

# %%
#  Summary of target variable 
print(y.describe())

# %%
#  Distribution of Exam Score 
sns.kdeplot(y)
plt.title("Distribution of Exam Scores")
plt.xlabel("Exam Score")
plt.show()

# %%
# Relationship: Hours Studied vs Exam Score 
plt.scatter(df["Hours_Studied"], df["Exam_Score"])
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.title("Hours Studied vs Exam Score")
plt.show()

# %%
#  Relationship: Attendance vs Exam Score 
plt.scatter(df["Attendance"], df["Exam_Score"])
plt.xlabel("Attendance")
plt.ylabel("Exam Score")
plt.title("Attendance vs Exam Score")
plt.show()
# %%
# ANSWER: 
# I cleaned the dataset by removing observations with missing values in variables such as Teacher_Quality, Parental_Education_Level, and Distance_from_Home. This ensures that the dataset is complete and suitable for regression analysis. I also converted all categorical variables into dummy variables using one-hot encoding so they can be used in a linear regression model.
# I selected Exam_Score as the target variable, since it represents student academic performance. The predictor variables include factors such as Hours_Studied, Attendance, Sleep_Hours, Previous_Scores, and other demographic and behavioral characteristics.

# From the summary statistics, Exam_Score has a mean of about 67.25 with a standard deviation of about 3.91, indicating that most students’ scores are relatively close to the average. The distribution of exam scores appears approximately bell-shaped, with most values concentrated between about 65 and 70, and a few higher outliers.

# The scatter plot of Hours_Studied versus Exam_Score shows a positive relationship, where students who study more hours tend to have higher exam scores. Similarly, the scatter plot of Attendance versus Exam_Score also shows a positive relationship, indicating that higher attendance is associated with better performance.
#Overall, the exploratory analysis suggests that both study time and attendance are important predictors of exam scores, and the selected variables are appropriate for modeling student performance using linear regression.
# %%
#3.Split the sample into an ~80% training set and a ~20% test set.

# 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(  X, y, test_size=0.2, random_state=42)

# Check sizes
print("Training set size:", X_train.shape[0]) #51
print("Test set size:", X_test.shape[0]) #1276

#I split the dataset into an 80% training set and a 20% test set using the train_test_split function. The training set is used to fit the linear regression model, while the test set is used to evaluate how well the model performs on unseen data.

#I used a random_state of 42 to ensure that the results are reproducible.
# %%
#4. Run a few regressions of your target/outcome variable on a variety of features/predictors. 
# Compute the RMSE on the test set.

# %%
# Model 1: Numeric variables only 
X_num = X[["Hours_Studied", "Attendance", "Sleep_Hours", "Previous_Scores"]]

# Split again to match features
Xn_train, Xn_test, yn_train, yn_test = train_test_split(
    X_num, y, test_size=0.2, random_state=42
)

model1 = LinearRegression()
model1.fit(Xn_train, yn_train)

y_pred1 = model1.predict(Xn_test)

rmse1 = np.sqrt(mean_squared_error(yn_test, y_pred1))
print("Model 1 RMSE (numeric only):", rmse1)

# %%
#Model 2: Key predictors 
X_key = X[["Hours_Studied", "Attendance", "Previous_Scores"]]

Xk_train, Xk_test, yk_train, yk_test = train_test_split(
    X_key, y, test_size=0.2, random_state=42
)

model2 = LinearRegression()
model2.fit(Xk_train, yk_train)

y_pred2 = model2.predict(Xk_test)

rmse2 = np.sqrt(mean_squared_error(yk_test, y_pred2))
print("Model 2 RMSE (key variables):", rmse2)

# %%
# Model 3: All variables 
model3 = LinearRegression()
model3.fit(X_train, y_train)

y_pred3 = model3.predict(X_test)

rmse3 = np.sqrt(mean_squared_error(y_test, y_pred3))
print("Model 3 RMSE (all variables):", rmse3)

# %%

# ANSWER: 
# I ran several linear regression models using different sets of predictors to evaluate how well they predict Exam_Score.

#The first model used only numeric variables such as Hours_Studied, Attendance, Sleep_Hours, and Previous_Scores. This model had an RMSE of approximately 2.55.

#The second model used a smaller set of key predictors, including Hours_Studied, Attendance, and Previous_Scores, with an RMSE of approximately 2.55 as well.

#The third model used all available predictors, including both numeric variables and dummy-coded categorical variables. This model had the lowest RMSE of approximately 2.04, indicating the best performance.

#Overall, the model that included all variables performed the best. This suggests that including additional factors such as demographic and behavioral variables improves the model’s ability to predict exam scores, even though the improvement is moderate.

# %%
#5. Which model performed the best, and why?

# ANSWER: 
# The model that performed the best was the third model, which included all available predictors, including both numeric variables and dummy-coded categorical variables. This model had the lowest RMSE of approximately 2.04, compared to about 2.55 for the other models.

# This model performs better because it incorporates more information about each student, including demographic, behavioral, and school-related factors. By including these additional variables, the model is able to better capture the variation in exam scores and make more accurate predictions.

# In contrast, the models that used only numeric variables or a smaller subset of predictors had higher RMSE values, indicating less accurate predictions. Therefore, including a wider range of relevant features improves model performance.
# %%
#6. What did you learn?

# ANSWER: 
# Through this analysis, I learned how powerful even simple linear regression models can be when they are built thoughtfully. 
# At the beginning, I assumed that just using a few obvious numeric variables like hours studied or attendance would be enough to predict exam scores, but I saw that those models were limited.
# What really stood out to me was how much better the model performed when I included categorical variables like motivation level, school type, and family background. 
# It made me realize that outcomes like academic performance are influenced by more than just numbers, and that context and environment matter a lot.
# I also learned the importance of comparing models rather than assuming one approach is best. 
# Even small differences in RMSE helped me see which variables actually improved predictions. 
# Overall, this helped me better understand how to build, evaluate, and interpret regression models, and it made the process feel much more intuitive.
# %%
