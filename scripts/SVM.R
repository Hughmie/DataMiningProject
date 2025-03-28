library(dplyr)
library(e1071)
library(pROC)
library(missForest)


# load data
data <- read.csv("datasets/data_outliers_processed.csv", stringsAsFactors = FALSE)


# inspect data
glimpse(data)
sapply(data, unique) # there are missing values "unknown"


# categorical -> factor
# foctor -> dummy can be done automatically in svm()
cate_vars <- c("job","marital","education","default",
                            "housing","loan","contact","month",
                            "day_of_week","poutcome", "y")
data[cate_vars] <- lapply(data[cate_vars], factor)
glimpse(data)


# process missing values with missforest(after categorical -> factor)
sapply(data, function(x) sum(x == "unknown"))
data[data == "unknown"] <- NA

set.seed(123)
data_filled <- missForest(data)
data <- data_filled$ximp


# split data into train/test/val (0.5:0.25:0.25)
set.seed(111)
n <- nrow(data)
train_idx <- sample(1:n, 0.5 * n)
remain_idx <- setdiff(1:n, train_idx)
test_idx <- sample(remain_idx, 0.5 * length(remain_idx))
val_idx <- setdiff(remain_idx, test_idx)

train_data <- data[train_idx, ]
test_data <- data[test_idx, ]
val_data <- data[val_idx, ]


# standardize numeric variables with z-score
num_vars <- c("age","campaign","pdays","previous",
              "emp.var.rate","cons.price.idx","cons.conf.idx",
              "euribor3m","nr.employed")

# z-score scaling (to avoid data leakage, μ and σ can't be calculated in test and val sets)
train_data[num_vars] <- scale(train_data[num_vars])

means <- colMeans(train_data[num_vars], na.rm = TRUE)
sds <- apply(train_data[num_vars], 2, sd, na.rm = TRUE)

test_data[num_vars] <- scale(test_data[num_vars],
                             center = means,
                             scale = sds)

val_data[num_vars] <- scale(val_data[num_vars],
                            center = means,
                            scale = sds)

summary(rbind(train_data, val_data)$y)


# shuffle the train_data
train_data <- train_data[sample(1:nrow(train_data)), ]


# == Linear Kernel ==


# automatic hyperparameter tuning
# as turning is autometic, val_data is not necessary
set.seed(222)
tune_linear <- tune(svm, 
                    y ~ ., 
                    data = rbind(train_data, val_data), 
                    kernel = "linear", 
                    scale = FALSE,
                    ranges = list(cost = c(0.001, 0.01, 0.1, 0.5, 1, 10, 50, 100)))
print(tune_linear)


# use the best cost to fit
best_cost_linear <- tune_linear$best.parameters$cost
# Unlike neural networks, 
# random seeds are not required for training, 
# as there is no random initialization
final_model_linear <- svm(y ~ ., 
                          data = rbind(train_data, val_data), 
                          kernel = "linear",
                          cost = best_cost_linear, 
                          probability = TRUE, 
                          scale = FALSE,
                          class.weights = c("no" = 1, "yes" = 5), 
                          control = list(maxiter = 500)
                          )


# == RBF Kernel ==


set.seed(333)
tune_rbf <- tune(svm, 
                 y ~ ., 
                 data = train_data, 
                 kernel = "radial", 
                 scale = FALSE,
                 ranges = list(cost = c(0.001, 0.01, 0.1, 0.5, 1, 10, 50, 100),
                               gamma = c(0.001, 0.01, 0.1, 0.5, 1, 10)))
print(tune_rbf)


# use the best cost and gamma to fit
best_cost_rbf <- tune_rbf$best.parameters$cost
best_gamma_rbf <- tune_rbf$best.parameters$gamma
final_model_rbf <- svm(y ~ ., 
                       data = rbind(train_data, val_data), 
                       kernel = "radial",
                       cost = best_cost_rbf, 
                       gamma = best_gamma_rbf,
                       probability = TRUE, 
                       scale = FALSE,
                       class.weights = c("no" = 1, "yes" = 5),
                       control = list(maxiter = 500)
                       )


# == Prediction ==


# prediction of linear kernel
pred_linear <- predict(final_model_linear, test_data) # it's a vector in the same length as nrow(test_data)


# confusion matrix of linear kernel
conf_mat_linear <- table(Actual = test_data$y, 
                         Predicted = pred_linear)
conf_mat_linear


# calculate the accuracy, sensitivity and specificity 
accuracy_linear <- sum(diag(conf_mat_linear)) / sum(conf_mat_linear)
sensitivity_linear <- conf_mat_linear[2,2] / sum(conf_mat_linear[2,])
specificity_linear <- conf_mat_linear[1,1] / sum(conf_mat_linear[1,])
cat("Accuracy(linear): ", accuracy_linear, "\n")
cat("Sensitivity(linear): ", sensitivity_linear, "\n")
cat("Specificity(linear): ", specificity_linear, "\n")


# prediction of RBF kernel
pred_rbf <- predict(final_model_rbf, test_data)


# confusion matrix of RBF kernel
conf_mat_rbf <- table(Actual = test_data$y, 
                         Predicted = pred_rbf)
conf_mat_rbf


# calculate the accuracy, sensitivity and specificity 
accuracy_rbf <- sum(diag(conf_mat_rbf)) / sum(conf_mat_rbf)
sensitivity_rbf <- conf_mat_rbf[2,2] / sum(conf_mat_rbf[2,])
specificity_rbf <- conf_mat_rbf[1,1] / sum(conf_mat_rbf[1,])
cat("Accuracy(RBF): ", accuracy_rbf, "\n")
cat("Sensitivity(RBF): ", sensitivity_rbf, "\n")
cat("Specificity(RBF): ", specificity_rbf, "\n")


# extract the probabilities of "yes"
pred_prob_linear <- attr(predict(final_model_linear, test_data, probability = TRUE), "probabilities")[, "yes"]
pred_prob_rbf <- attr(predict(final_model_rbf, test_data, probability = TRUE), "probabilities")[, "yes"]

# ROC and AUC
roc_linear <- roc(test_data$y, pred_prob_linear)
roc_rbf <- roc(test_data$y, pred_prob_rbf)
cat("AUC(linear): ", auc(roc_linear), "\n")
cat("AUC(RBF): ", auc(roc_rbf), "\n")

plot(roc_linear, col = "blue", main = "ROC of linear/RBF")
lines(roc_rbf, col = "red")
legend("bottomright", 
       legend = c(paste("Linear SVM, AUC =", auc(roc_linear)),
                  paste("RBF SVM, AUC =", auc(roc_rbf))),
       col = c("blue", "red"), lwd = 2)
