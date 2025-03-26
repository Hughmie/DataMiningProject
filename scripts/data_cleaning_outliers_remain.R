library(dplyr)

# load data
data <- read.csv("datasets\\group_23.csv")

# inspect data
glimpse(data)
head(data)
summary(data)
sapply(data, unique)
colSums(is.na(data)) # no NA

# count the number of missing values: 'unknown'
colSums(data == 'unknown')

# remove 'duration'
data$duration <- NULL

write.csv(data, "datasets\\data_outliers_remain.csv")
