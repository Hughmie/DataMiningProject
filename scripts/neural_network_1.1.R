library(neuralnet)
library(dplyr)
library(NeuralNetTools)
library(pROC)


#import data
data <- read.csv("datasets/data_outliers_processed.csv", stringsAsFactors = FALSE)


# inspect data
glimpse(data)


# categorical -> factor (except y)
# prepare for the conversion to dummy
cate_vars <- c("job","marital","education","default",
               "housing","loan","contact","month",
               "day_of_week","poutcome")
data[cate_vars] <- lapply(data[cate_vars], factor)
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
                             day_of_week + poutcome - 1, data = test_data)


# after the process of categorical variables, 
# then turn to numeric


# min-max normalization
num_vars <- c("age","campaign","pdays","previous",
              "emp.var.rate","cons.price.idx","cons.conf.idx",
              "euribor3m","nr.employed")
train_num <- train_data[, num_vars]
test_num <- test_data[, num_vars]
maxs <- apply(train_num, 2, max)
mins <- apply(train_num, 2, min)
train_num_scaled <- scale(train_num, center = mins, scale = maxs - mins)
test_num_scaled  <- scale(test_num,  center = mins, scale = maxs - mins)


# merge the processed data: dummy, num_scaled and y
train_processed <- data.frame(as.data.frame(train_dummy),
                              as.data.frame(train_num_scaled),
                              y = as.numeric(ifelse(train_data$y == 'yes', 1, 0)))
test_processed <- data.frame(as.data.frame(test_dummy),
                             as.data.frame(test_num_scaled),
                             y = as.numeric(ifelse(test_data$y == 'yes', 1, 0)))


# inspect the distribution of y in train_processed
table(train_processed$y) # there is an imbalance between 'yes' and 'no'


# as the imbalance in outputs, use down-sampling
# separate the training samples into positive and negative
positive_samples <- train_processed[train_processed$y == 1, ]
negative_samples <- train_processed[train_processed$y == 0, ]


# set the ratio of sampling
n_positive_sampling <- nrow(positive_samples)
sampling_ratio <- 3
n_negative_sampling <- n_positive_sampling * sampling_ratio


# randomly select negative samples in training data
set.seed(222)
negative_indices_to_keep <- sample(1:nrow(negative_samples), n_negative_sampling)
negative_samples_downsampled <- negative_samples[negative_indices_to_keep, ]

# merge the positive/negative samples after down-sampling
train_balanced <- rbind(positive_samples, negative_samples_downsampled)

# ramdonly order the data
train_balanced <- train_balanced[sample(1:nrow(train_balanced)), ]


# inspect the distribution of y in training set again
table(train_balanced$y)


# build and train neural network
set.seed(333)
nn_model <- neuralnet(
  y ~ ., 
  data = train_balanced, 
  hidden = c(20, 10, 5),
  act.fct = "logistic",
  linear.output = FALSE,
  err.fct = "ce"
)


# test on test_processed
test_set <- subset(test_processed, select = -y) # truncate output and only use inputs
pred_result <- compute(nn_model, test_set) # forward propagation
pred_prob <- pred_result$net.result
head(pred_prob)


# plot ROC and calculate AUC to find the best threshold (better than the fixed value 0.5)
roc_curve <- roc(test_processed$y, pred_prob)
plot(roc_curve, main = 'ROC curve')
auc(roc_curve)


# find the best threshold
# based on Youden index 
# i.e. maximise (sensitivity + specificity - 1)
coords <- coords(roc_curve, "best", ret = c("threshold", "sensitivity", "specificity"))
best_threshold <- coords[1, "threshold"]
best_sensitivity <- coords[1, "sensitivity"]
best_specifity <- coords[1, "specificity"]
cat("best threshold: ", round(best_threshold, 4), "\n")
cat("best sensitivity: ", round(best_sensitivity, 4), "\n")
cat("best specifity: ", round(best_specifity, 4), "\n")


# use the best threshold to project probabilities to 0/1
pred_class <- ifelse(pred_prob > best_threshold, 1, 0)


# build confusion matrix
conf_mat <- table(Actual = test_processed$y, Predicted = pred_class)
conf_mat


# calculate the accuracy, sensitivity and specificity 
accuracy <- sum(diag(conf_mat)) / sum(conf_mat)
sensitivity <- conf_mat[2,2] / sum(conf_mat[2,])
specificity <- conf_mat[1,1] / sum(conf_mat[1,])
cat("Accuracy: ", accuracy, "\n")
cat("Sensitivity: ", sensitivity, "\n")
cat("Specificity: ", specificity, "\n")