# coding: utf-8

library(randomForest)  # 随机森林算法
library(caret)        # 数据分割和模型评估
library(ggplot2)      # 数据可视化
library(dplyr)        # 数据操作
library(ROCR)         # ROC曲线绘制

data <- read.csv("group_23.csv", stringsAsFactors = TRUE)
data <- data %>% select(-duration)
data <- data %>% select(-pdays)
# 查看数据结构
str(data)

# 将目标变量转换为因子
data$y <- as.factor(data$y)

# 检查类别分布（处理不平衡数据）
table(data$y)
prop.table(table(data$y))

set.seed(0)

# 创建训练集和测试集
train_Index <- createDataPartition(data$y, p = 0.7, list = FALSE)
train_Data <- data[trainIndex, ]
test_Data <- data[-trainIndex, ]

# 检查分割后的类别分布
table(train_Data$y)
table(test_Data$y)

rf_model <- randomForest(y ~ ., 
                         data = train_Data,
                         ntree = 149,          # 树的数量
                         mtry = sqrt(ncol(train_Data) - 1),  # 每棵树使用的特征数
                         importance = TRUE,    # 计算特征重要性
                         proximity = TRUE,     # 计算样本相似度
                         strata = train_Data$y, # 用于分层抽样
                         sampsize = c("no" = sum(train_Data$y == "no"), 
                         "yes" = sum(train_Data$y == "yes"))) # 平衡样本大小
              
  
# 查看模型摘要
print(rf_model)
  
predictions <- predict(rf_model, test_Data)
  
# 混淆矩阵
confusionMatrix(predictions, test_Data$y,positive = "yes")
  
# 计算预测概率（用于ROC曲线）
predict_prob <- predict(rf_model, test_Data, type = "prob")[, "yes"]
  
# 创建预测对象
pred <- prediction(predict_prob, test_Data$y)
  
# 计算性能指标
perf <- performance(pred, "tpr", "fpr")
  
# 绘制ROC曲线
plot(perf, colorize = TRUE, main = "ROC Curve for Random Forest")
abline(a = 0, b = 1, lty = 2)
  
# 计算AUC值
auc <- performance(pred, "auc")@y.values[[1]]
cat("AUC:", auc, "\n")
  
  