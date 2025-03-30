# coding: utf-8

# Load packages
library(rpart)       
library(rpart.plot)  
library(caret)       
library(dplyr)
library(ROSE)
library(pROC) 

# Read data
data <- read.csv("group_23.csv", stringsAsFactors = TRUE)
# Remove duration column
data <- data %>% select(-duration)
data <- data %>% select(-pdays)
# Check missing values
col_na <- colSums(is.na(data))
print("Number of missing values per column:")
print(colSums(is.na(data)))

# IQR Outlier Detection
outlier_iqr <- function(x) {
  Q <- quantile(x, probs=c(0.25, 0.75), na.rm=TRUE)
  iqr <- IQR(x, na.rm=TRUE)
  lower <- Q[1] - 1.5*iqr
  upper <- Q[2] + 1.5*iqr
  x < lower | x > upper
}

outliers <- outlier_iqr(data$campaign)
max(data$campaign)

# Check data structure
str(data)

### Data Processing
# Convert y column to factor (for classification problems)
data$y <- as.factor(data$y)

# Check class distribution
table(data$y)

# Split data into training and test sets
set.seed(0) # Ensure reproducibility
train_index <- createDataPartition(data$y, p = 0.7, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

# Oversampling
balanced_train_data <- ovun.sample(y ~ ., data = train_data, method = "over", N = 12410)$data
table(balanced_train_data$y)
# loss_matrix <- matrix(c(0, 1, 5, 0), nrow = 2)

# Build decision tree model
tree_model <- rpart(y ~ ., 
                    data = balanced_train_data, 
                    method = "class", # For classification
                    parms = list(split = "information"), # Use information gain
                    control = rpart.control(minsplit = 20, # Minimum samples per node
                                            minbucket = 200,  # Minimum samples per leaf
                                            maxdepth = 5,   # Maximum tree depth
                                            cp = 0.001))     # Complexity parameter
test_pred <- predict(tree_model, test_data, type = "class")
confusionMatrix(test_pred, test_data$y, positive = "yes")

# View model summary
print(tree_model)
summary(tree_model)

# Visualize decision tree
rpart.plot(tree_model, 
           type = 4, 
           extra = 104, 
           box.palette = "GnBu", 
           branch.lty = 3, 
           shadow.col = "gray", 
           nn = TRUE)

# Evaluate model on training set
train_pred <- predict(tree_model, train_data, type = "class")
confusionMatrix(train_pred, train_data$y, positive = "yes")

# Evaluate model on test set
test_pred <- predict(tree_model, test_data, type = "class")
confusionMatrix(test_pred, test_data$y, positive = "yes")

test_pred_prob <- predict(tree_model, test_data, type = "prob")[, "yes"]
roc_obj <- roc(test_data$y, test_pred_prob, levels = c("no", "yes"))

# Plot ROC curve
plot(roc_obj, 
     main = "ROC Curve for Decision Tree Model",
     col = "blue",
     lwd = 2,
     print.auc = TRUE,
     auc.polygon = TRUE,
     auc.polygon.col = "lightblue",
     print.thres = TRUE)

# Calculate AUC value
auc_value <- auc(roc_obj)
cat("AUC value:", auc_value, "\n")

# View variable importance
tree_model$variable.importance

# Plot CP table for pruning
plotcp(tree_model)

# Prune tree (if needed)
# pruned_tree <- prune(tree_model, cp = tree_model$cptable[which.min(tree_model$cptable[,"xerror"]),"CP"])
# rpart.plot(pruned_tree)

test_pred <- predict(tree_model, test_data, type = "prob")[, "yes"]
roc_obj <- roc(test_data$y, test_pred, levels = c("no", "yes"))

# Plot ROC curve
plot(roc_obj, 
     main = "ROC Curve for Decision Tree Model",
     col = "blue",
     lwd = 2,
     print.auc = TRUE,
     auc.polygon = TRUE,
     auc.polygon.col = "lightblue",
     print.thres = TRUE)


