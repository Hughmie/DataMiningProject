library(dplyr)
library(ggplot2)

# load data
data <- read.csv("datasets\\data_outliers_remain.csv")

# find outliers
num_cols <- sapply(data, is.numeric) # find out which columns are numeric

find_outliers <- function(x) {
  Q1 <- quantile(x, 0.25)
  Q3 <- quantile(x, 0.75)
  IQR <- Q3 - Q1
  lower <- Q1 - 1.5 * IQR
  upper <- Q3 + 1.5 * IQR
  
  sum(x < lower | x > upper) # count the number of outliers in each column
}
sapply(data[, num_cols], find_outliers)

summary(data)

# After analysis of the data, 
# it's only necessary to process the outliers of 'campaign'

# boxplot of 'campaign'
ggplot(data, aes(y = campaign)) + 
  geom_boxplot(outlier.colour = "red") + 
  labs(title = "Distribution of Campaign",
       y = "Number of Contacts",
       x = "")

# histogram of 'campaign'
ggplot(data, aes(x = campaign)) + 
  geom_histogram() + 
  labs(title = "Histogram of Campaign", 
       x = "Number of Contacts", 
       y = "")

# Truncate severe outliers from 'campaign' using Winsorizing
Q3 <- quantile(data$campaign, 0.75)
Q1 <- quantile(data$campaign, 0.25)
IQR <- Q3 - Q1
severe_upper <- Q3 + 3 * IQR
print(severe_upper)
data$campaign[data$campaign > severe_upper] <- severe_upper
summary(data$campaign)

# save data
write.csv(data, "datasets\\data_outliers_processed.csv")
