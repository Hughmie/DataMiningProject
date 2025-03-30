# coding: utf-8

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

  
