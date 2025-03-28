library(dplyr)
library(class)
library(missForest)
library(pROC)


# import data
data <- read.csv("datasets/data_outliers_processed.csv", stringsAsFactors = FALSE)


# inspect data
glimpse(data)
sapply(data, unique) # there are missing values "unknown"

# categorical -> factor
cate_vars <- cate_vars <- c("job","marital","education","default",
                            "housing","loan","contact","month",
                            "day_of_week","poutcome", "y")
data[cate_vars] <- lapply(data[cate_vars], factor)
glimpse(data)


# process missing values with missforest
colSums(data == "unknown")
data["unknown"] <- NA

set.seed(123)
data_filled <- missForest(data)
data <- data_filled$ximp


# train-test random splitting(0.75 : 0.25), 
# as the use of cross-validation, val_set is not necessary
set.seed(111)
n <- nrow(data)
train_idx <- sample(1:n, 0.75 * n)
test_idx <- setdiff(1:n, train_idx)

train_data <- data[train_idx, ]
test_data <- data[test_idx, ]

dim(train_data)
dim(test_data)


# factor -> dummy
# model.matrix() will create a new matrix without other variables that aren't mentioned
train_dummy <- model.matrix(~ job + marital + education + default + 
                              housing + loan + contact + month + 
                              day_of_week + poutcome - 1, data = train_data)
test_dummy <- model.matrix(~ job + marital + education + default + 
                             housing + loan + contact + month + 
                             day_of_week + poutcome - 1, data = test_data)

head(train_dummy)


# standardize numeric variables with z-score
num_vars <- c("age","campaign","pdays","previous",
              "emp.var.rate","cons.price.idx","cons.conf.idx",
              "euribor3m","nr.employed")

train_num <- train_data[, num_vars]
test_num <- test_data[, num_vars]

train_num_scaled <- scale(train_num)

means <- colMeans(train_num, na.rm = TRUE)
sds <- apply(train_num, 2, sd, na.rm = TRUE)
means
sds

test_num_scaled <- scale(test_num, 
                         center = means, 
                         scale = sds
                         )


# merge the processed data: dummy, num_scaled and y
train_data <- data.frame(as.data.frame(train_dummy),
                              as.data.frame(train_num_scaled),
                              y = as.numeric(ifelse(train_data$y == 'yes', 1, 0)))
test_data <- data.frame(as.data.frame(test_dummy),
                             as.data.frame(test_num_scaled),
                             y = as.numeric(ifelse(test_data$y == 'yes', 1, 0)))

dim(train_data)


# == k-fold cross-validation ==


# define n_folds and k_value
n_fold <- 10
k_value <- 1:10
feature_cols_idx <- setdiff(colnames(train_data), "y") # extract the idx of features

sample_fold_idx <- sample(rep(1:n_fold, length.out = nrow(train_data))) # allocate a fold idx for each sample, then shuffle
head(sample_fold_idx, 10)
table(sample_fold_idx)

results <- matrix(0, nrow = length(k_value), ncol = n_fold) # save the accuracy of each k and fold

for(k_idx in 1:length(k_value)) {
  k <- k_value[k_idx]
  for(fold in 1:n_fold) {
    # devide the training set and validation set of the current fold
    train_fold <- train_data[sample_fold_idx != fold, ]
    val_fold <- train_data[sample_fold_idx == fold, ]
    
    # predict with current k
    pred <- knn(train = train_fold[, feature_cols_idx], 
                test = val_fold[, feature_cols_idx], 
                cl = train_fold$y, 
                k = k)
    
    # calculate and save current accuracy
    results[k_idx, fold] <- mean(pred == val_fold$y)
  }
}

# calculate the mean accuracy of each k
k_accuracy <- rowMeans(results)

# find the best k and accuracy
best_k <- k_value[which.max(k_accuracy)]
best_accuracy <- max(k_accuracy)
cat("The best k is ", best_k, "\n")
cat("Its accuracy of prediction is ", best_accuracy, "\n")


# test
final_pred <- knn(train = train_data[, feature_cols_idx], 
                         test = test_data[, feature_cols_idx],
                         cl = train_data$y, 
                         k = best_k, 
                         prob = TRUE)


# extract the probabilities of final_pred
prob_values <- attr(final_pred, "prob")


# convert all the probilities to prediction probabilities of 1
prob_positive <- ifelse(final_pred == 1, prob_values, 1 - prob_values)


# ROC and AUC
roc <- roc(test_data$y, prob_positive)
auc <- auc(roc)

plot(roc, main = "ROC")
legend("bottomright", 
       legend = paste("AUC = ", auc))


# confusion matrix
conf_mat <- table(Actual = test_data$y, Predicted = final_pred)
conf_mat


# calculate the accuracy, sensitivity and specificity 
accuracy <- sum(diag(conf_mat)) / sum(conf_mat)
sensitivity <- conf_mat[2,2] / sum(conf_mat[2,])
specificity <- conf_mat[1,1] / sum(conf_mat[1,])
cat("Accuracy: ", accuracy, "\n")
cat("Sensitivity: ", sensitivity, "\n")
cat("Specificity: ", specificity, "\n")
