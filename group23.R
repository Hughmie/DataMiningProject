# coding: utf-8





########################
######decision tree#####

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

######################
#####randomForest#####

library(randomForest)  
library(caret)       
library(ggplot2)      
library(dplyr)        
library(ROCR)         

# Load and preprocess data
data <- read.csv("group_23.csv", stringsAsFactors = TRUE)
data <- data %>% select(-duration)
data <- data %>% select(-pdays)

# Check data structure
str(data)

# Convert target variable to factor
data$y <- as.factor(data$y)

# Check class distribution (Handle imbalanced data)
table(data$y)
prop.table(table(data$y))

set.seed(0)

# Create training and test sets
train_Index <- createDataPartition(data$y, p = 0.7, list = FALSE)
train_Data <- data[train_Index, ]
test_Data <- data[-train_Index, ]

# Check class distribution after splitting
table(train_Data$y)
table(test_Data$y)

# Build Random Forest model
rf_model <- randomForest(y ~ ., 
                         data = train_Data,
                         ntree = 149,          # Number of trees
                         mtry = sqrt(ncol(train_Data) - 1),  # Features per tree
                         importance = TRUE,    # Calculate feature importance
                         proximity = TRUE,     # Calculate sample proximity
                         strata = train_Data$y, # Stratified sampling
                         sampsize = c("no" = sum(train_Data$y == "no"), 
                                      "yes" = sum(train_Data$y == "yes"))) # Balance sample sizes

# Display model summary
cat("\nRandom Forest Model Summary:\n")
print(rf_model)

# Generate predictions
predictions <- predict(rf_model, test_Data)

# Evaluate performance with confusion matrix
cat("\nConfusion Matrix:\n")
print(confusionMatrix(predictions, test_Data$y, positive = "yes"))

# Calculate class probabilities for ROC analysis
predict_prob <- predict(rf_model, test_Data, type = "prob")[, "yes"]

# Create ROC object
roc_obj <- roc(
  response = test_Data$y,
  predictor = predict_prob,
  levels = c("no", "yes")
)

# Plot ROC curve with enhanced visualization
plot(roc_obj, 
     main = "ROC Curve for Random forest Model",
     col = "blue",
     lwd = 2,
     print.auc = TRUE,
     auc.polygon = TRUE,
     auc.polygon.col = "lightblue",
     print.thres = TRUE)

# Add reference line


# Display AUC with confidence interval
cat("\nModel Performance Metrics:")
cat("\nAUC:", round(auc(roc_obj), 3))
cat("\n95% CI:", round(ci(roc_obj), 3))

#######################
#####XGBoost model#####
library(xgboost)    # XGBoost model
library(caret)      # Data splitting and preprocessing
library(dplyr)      # Data manipulation
library(pROC)       # ROC curve evaluation

# 2. Data Preparation and Preprocessing
# Read data
data <- read.csv("group_23.csv", stringsAsFactors = TRUE)

# Remove duration column
data <- data %>% select(-duration)
data <- data %>% select(-pdays)
# Check missing values

# Convert target variable to 0/1
data$y <- ifelse(data$y == "yes", 1, 0)

# 3. Handle Categorical Variables (without Matrix package)
# Method 1: Label Encoding
# Convert factor variables to numeric
data <- data %>%
  mutate(across(where(is.factor), as.numeric))

# OR Method 2: Manual dummy variables (without Matrix)
# Here we choose Method 1 for simplicity and tree model compatibility

# 4. Split Dataset
set.seed(0)  # Ensure reproducibility
train_index <- createDataPartition(data$y, p = 0.7, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

# 5. Prepare XGBoost Input Data
# Separate features and target variable
train_x <- as.matrix(train_data %>% select(-y))
train_y <- train_data$y
test_x <- as.matrix(test_data %>% select(-y))
test_y <- test_data$y

# Convert to XGBoost DMatrix format
dtrain <- xgb.DMatrix(data = train_x, label = train_y)
dtest <- xgb.DMatrix(data = test_x, label = test_y)

# 6. Set XGBoost Parameters
params <- list(
  objective = "binary:logistic",  # Binary classification
  eval_metric = "logloss",        # Evaluation metric
  max_depth = 5,                  # Maximum tree depth
  eta = 0.3,                      # Learning rate
  subsample = 0.8,                # Sample subsampling ratio
  colsample_bytree = 0.8,         # Feature subsampling ratio
  min_child_weight = 1            # Minimum sum of instance weight
)

# 7. Train Model
xgb_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 150,                  # Number of iterations
  watchlist = list(train = dtrain, test = dtest),
  early_stopping_rounds = 10,     # Early stopping rounds
  print_every_n = 10              # Print frequency
)

# Evaluate Model (using confusionMatrix)
test_pred_prob <- predict(xgb_model, dtest)
test_pred <- factor(ifelse(test_pred_prob > 0.5, "yes", "no"), 
                    levels = c("no", "yes"))
test_actual <- factor(ifelse(test_data$y == 1, "yes", "no"), 
                      levels = c("no", "yes"))

confusionMatrix(test_pred, test_actual, positive = "yes")

# ROC Curve
roc_obj <- roc(test_actual, test_pred_prob)

plot(roc_obj, 
     main = "ROC Curve for Xgboost Model",
     col = "blue",
     lwd = 2,
     print.auc = TRUE,
     auc.polygon = TRUE,
     auc.polygon.col = "lightblue",
     print.thres = TRUE)

