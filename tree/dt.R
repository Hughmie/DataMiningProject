# coding: utf-8

# 加载包
library(rpart)       # 用于构建决策树
library(rpart.plot)  # 用于可视化决策树
library(caret)       # 用于模型评估和数据处理
library(dplyr)
library(ROSE)

# 读取数据
data <- read.csv("group_23.csv", stringsAsFactors = TRUE)
#删除 duration列
data <- data %>% select(-duration)
data <- data %>% select(-pdays)
# 检查缺失值
col_na <- colSums(is.na(data))
print("各列缺失值数量:")
print(colSums(is.na(data)))

# 四分位异常值
outlier_iqr <- function(x) {
  Q <- quantile(x, probs=c(0.25, 0.75), na.rm=TRUE)
  iqr <- IQR(x, na.rm=TRUE)
  lower <- Q[1] - 1.5*iqr
  upper <- Q[2] + 1.5*iqr
  x < lower | x > upper
}

outliers <- outlier_iqr(data$campaign)
max(data$campaign)



# 检查数据结构
str(data)

### 数据处理
# 将y列转换为因子（如果是分类问题）
data$y <- as.factor(data$y)


#for(col in names(data)){
#  cat("\nColumn:", col, "\n")
#  print(table(data[[col]]))
#}

# 检查类别分布
table(data$y)

# 分割数据为训练集和测试集
set.seed(0) # 确保结果可重现
train_index <- createDataPartition(data$y, p = 0.7, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

# 超采样
balanced_train_data <- ovun.sample(y ~ ., data = train_data, method = "over", N = 12410)$data
table(balanced_train_data$y)
#loss_matrix <- matrix(c(0, 1, 5, 0), nrow = 2)


# 构建决策树模型
tree_model <- rpart(y ~ ., 
                    data = balanced_train_data, 
                    method = "class", # 用于分类
                    parms = list(split = "information"), # 使用信息增益
                    control = rpart.control(minsplit = 20, # 节点最小样本数
                                            minbucket = 200,  # 叶节点最小样本数
                                            maxdepth = 5,   # 树的最大深度
                                            cp = 0.001))     # 复杂度参数
test_pred <- predict(tree_model, test_data, type = "class")
confusionMatrix(test_pred, test_data$y,positive = "yes")
#tree_model <- rpart(y ~ ., data = balanced_train_data, 
                      #method = "class",parms = list(split = "information", 
                      #loss = loss_matrix),control = rpart.control(minsplit = 5,
                      #minbucket = 2, maxdepth = 10, cp = 0.0001))







# 查看模型摘要
print(tree_model)
summary(tree_model)

# 可视化决策树
rpart.plot(tree_model, 
           type = 4, 
           extra = 104, 
           box.palette = "GnBu", 
           branch.lty = 3, 
           shadow.col = "gray", 
           nn = TRUE)

# 在训练集上评估模型
train_pred <- predict(tree_model, train_data, type = "class")
confusionMatrix(train_pred, train_data$y,positive = "yes")

# 在测试集上评估模型
test_pred <- predict(tree_model, test_data, type = "class")
confusionMatrix(test_pred, test_data$y,positive = "yes")

test_pred_prob <- predict(tree_model, test_data, type = "prob")[, "yes"]
roc_obj <- roc(test_data$y, test_pred_prob, levels = c("no", "yes"))

# 绘制ROC曲线
plot(roc_obj, 
     main = "ROC Curve for Decision Tree Model",
     col = "blue",
     lwd = 2,
     print.auc = TRUE,
     auc.polygon = TRUE,
     auc.polygon.col = "lightblue",
     print.thres = TRUE)

# 计算AUC值
auc_value <- auc(roc_obj)
cat("AUC值:", auc_value, "\n")


# 查看变量重要性
tree_model$variable.importance

# 绘制CP表格以选择最佳剪枝
plotcp(tree_model)

# 剪枝（如果需要）
# pruned_tree <- prune(tree_model, cp = tree_model$cptable[which.min(tree_model$cptable[,"xerror"]),"CP"])
# rpart.plot(pruned_tree)


test_pred <- predict(tree_model, test_data, type = "prob")[, "yes"]
roc_obj <- roc(test_data$y, test_pred, levels = c("no", "yes"))

# 绘制ROC曲线
plot(roc_obj, 
     main = "ROC Curve for Decision Tree Model",
     col = "blue",
     lwd = 2,
     print.auc = TRUE,
     auc.polygon = TRUE,
     auc.polygon.col = "lightblue",
     print.thres = TRUE)

# 计算AUC值
auc_value <- auc(roc_obj)
cat("AUC值:", auc_value, "\n")

