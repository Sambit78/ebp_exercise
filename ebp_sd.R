setwd("~/Google Drive/Data Analytics/Predictive HR Analytics/ebp_exercise")
#https://cran.r-project.org/web/packages/DataExplorer/vignettes/dataexplorer-intro.html
#https://www.mymarketresearchmethods.com/types-of-data-nominal-ordinal-interval-ratio/
#https://stats.idre.ucla.edu/r/dae/ordinal-logistic-regression/
#https://www.r-bloggers.com/how-to-perform-ordinal-logistic-regression-in-r/


# Libraries
library(corrplot)
library(tidyverse)
library(tidyquant)
library(readxl)
library(forcats)
library(stringr)
library(kableExtra)
library(skimr)
library(GGally)

#remove.packages("DataExplorer")
#install.packages("DataExplorer",dependencies = TRUE)


library(caret)

library(ggplot2)
library(ggthemes) 
library(gridExtra) 
library(vcd)

library(UsingR)
library(DataExplorer)


####
# 1. Data Preparation ----
####

final.data <- read.csv("data.csv",header = TRUE)
glimpse(perf.data)

# Returns variable types of the dataset
str(final.data)

# Provides information on missing values in the dataset
skim(final.data)

# Drop employee id column as it no impact to our modeling

perf.data <- drop_columns(final.data, c("employee_id"))

# Quickly visualize the data by using the plot_str function

perf.data %>%
  plot_str()

####
# 2. Exploratory Data Analysis ----
####

# Lets get introduced to our dataset 

introduce(perf.data)

# Visualize the data

plot_intro(perf.data)

# If fields showed missing data , you can visualize the missing data fields

plot_missing(perf.data)

# Return first 5 rows of the dataset

perf.data %>%
  head(5)%>%
  kable() %>%
  kable_styling(bootstrap_options = c("hover","striped","condensed"),full_width = F)

# Return %ages of workers in different classes of performance group

perf.data %>%
  group_by(performance_group) %>%
  summarise(n = n()) %>%
  ungroup() %>%
  mutate(pct = n / sum(n))%>%
  kable() %>%
  kable_styling(bootstrap_options = c("hover","striped","condensed"),full_width = F)

# This plot visualisizes frequency distributions for all the discrete features
plot_bar(perf.data)

# A bivirate frequency distribution will show the distribution of discrete features by Performance group

plot_bar(perf.data,with = "test_score")

# This plot visualises frequency distribution of all continous variables
# Customers , group size , yrs employed do not have a uniform distribution. Transfers data is mostly 0. Lets convert it to factor

plot_histogram(perf.data)
perf.data <- update_columns(perf.data,"transfers",as.factor)
perf.data %>%
  head(5)

# QQ plot is a way to visualise the deviation from a specific probability distribution. 
# customers , group_size and yrs_employed are skewed.

qq_data <- perf.data[, c("customers", "group_size", "test_score","yrs_employed")]
plot_qq(qq_data, sampled_rows = 1000L)

# Lets run the qq plot after converting the variables into the log values.

log_qq_data <- update_columns(qq_data, c("customers", "group_size", "yrs_employed"), function(x) log(x + 1))
plot_qq(log_qq_data[, c("customers", "group_size", "yrs_employed")], sampled_rows = 1000L)

# Lets transform the data in the perf.data table with the log values of the variables
perf.data <- update_columns(perf.data, c("customers", "group_size", "yrs_employed"), function(x) log(x + 1))

# Now lets visualize the QQ plot by the feature Performance_group

qq_data <- perf.data[, c("customers", "group_size", "yrs_employed", "test_score","performance_group")]
plot_qq(qq_data, by = "performance_group", sampled_rows = 1000L)


####
# 3. Correlation Analysis ----
####

# Plot correlation plot to display relationships
plot_correlation(na.omit(perf.data), maxcat = 5L)
# You can generate correlation plot for character and discrete variables separately also
plot_correlation(na.omit(perf.data), type = "c")
plot_correlation(na.omit(perf.data), type = "d")


####
# 4. Define Training and Test Dataset ----
####

#Dividing data into training and test set
#Random sampling 
samplesize = 0.80*nrow(perf.data)
set.seed(100)
index = sample(seq_len(nrow(perf.data)), size = samplesize)
#Creating training and test set 
datatrain = perf.data[index,]
datatest = perf.data[-index,]


####
# 5. Modelling - Ordinal Logistic Regression ----
####

# load car package
library(car)

# Use the Intercept Model as the initial model;

lower.rm <- polr(performance_group ~ 1,data=datatrain)
summary(lower.rm)

# Define upper model;
upper.rm <- polr(performance_group ~ .,data=datatrain)
summary(upper.rm)

####
# 5. Step AIC Model ----
####

# Step AIC model :

stepwise.rm <- stepAIC(upper.rm)


####
# 6. Collinearity ----
####
library(car)
sort(vif(lower.rm), decreasing = TRUE)
sort(vif(upper.rm),decreasing = TRUE)
sort(vif(stepwise.rm),decreasing = TRUE)


####
# 7. Feature Importance  ----
####

ctable <- coef(summary(stepwise.rm))
odds_ratio <- exp(coef(summary(stepwise.rm))[ , c("Value")])
p <- pnorm(abs(ctable[, "t value"]), lower.tail = FALSE) * 2
coef_summary <- cbind(ctable, as.data.frame(odds_ratio, nrow = nrow(ctable), ncol = 1), "p value" = p)

kable(coef_summary[1:(nrow(coef_summary) - 2), ]) %>%
kable_styling(bootstrap_options = c("hover","striped","condensed"),full_width = F)

####
# 8. Model Evaluation  ----
####

#Compute confusion table and misclassification error

predictperformance = predict(stepwise.rm,datatest[,-1])
table(datatest$performance_group, predictperformance)
mean(as.character(datatest$performance_group) != as.character(predictperformance))


