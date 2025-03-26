library(dplyr)

# import data
data <- read.csv("datasets/data_outliers_processed.csv", stringsAsFactors = FALSE)


# inspect data
glimpse(data)
table(data$y) # there is an imbalance between 'yes' and 'no'


# categorical -> factor (except y)
# prepare for the conversion to dummy
cate_vars <- c("job","marital","education","default",
               "housing","loan","contact","month",
               "day_of_week","poutcome")
data[cate_vars] <- lapply(data[cate_var], factor)
str(data)


# training-test random splitting (0.75:0.25)
set.seed(111)
index <- sample(1:nrow(data), round(0.75 * nrow(data)))

train_data <- data[index, ]
test_data <- data[-index, ]


# factor -> dummy
train_dummy <- model.matrix(~ job + marital + education + default + 
                              housing + loan + contact + month + 
                              day_of_week + poutcome - 1, data = train_data)
test_dummy <- model.matrix(~ job + marital + education + default + 
                             housing + loan + contact + month + 
                             day_of_week + poutcome - 1, data = train_data)


# after the process of categorial vaviables, 
# then turn to numerical
# min-max normalisation
num_vars <- c("age","campaign","pdays","previous",
              "emp.var.rate","cons.price.idx","cons.conf.idx",
              "euribor3m","nr.employed")
