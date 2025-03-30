# coding: utf-8

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
