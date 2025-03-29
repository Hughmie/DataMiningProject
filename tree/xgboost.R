# coding: utf-8


library(xgboost)    # XGBoost模型
library(caret)      # 数据分割和预处理
library(dplyr)      # 数据操作
library(pROC)       # ROC曲线评估

# 2. 数据准备和预处理
# 读取数据
data <- read.csv("group_23.csv", stringsAsFactors = TRUE)

#删除 duration列
data <- data %>% select(-duration)
data <- data %>% select(-pdays)
# 检查缺失值


# 将目标变量转换为0/1
data$y <- ifelse(data$y == "yes", 1, 0)

# 3. 处理分类变量（不使用Matrix包）
# 方法1：标签编码（Label Encoding）
# 将因子变量转换为数值
data <- data %>%
  mutate(across(where(is.factor), as.numeric))

# 或者方法2：手动创建哑变量（不使用Matrix）
# 这里我们选择方法1，因为它更简洁且适合树模型

# 4. 分割数据集
set.seed(0)  # 确保可重复性
train_index <- createDataPartition(data$y, p = 0.7, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

# 5. 准备XGBoost输入数据
# 分离特征和目标变量
train_x <- as.matrix(train_data %>% select(-y))
train_y <- train_data$y
test_x <- as.matrix(test_data %>% select(-y))
test_y <- test_data$y

# 转换为XGBoost的DMatrix格式
dtrain <- xgb.DMatrix(data = train_x, label = train_y)
dtest <- xgb.DMatrix(data = test_x, label = test_y)

# 6. 设置XGBoost参数
params <- list(
  objective = "binary:logistic",  # 二分类问题
  eval_metric = "logloss",        # 评估指标
  max_depth = 5,                  # 树的最大深度
  eta = 0.3,                      # 学习率
  subsample = 0.8,                # 样本采样比例
  colsample_bytree = 0.8,         # 特征采样比例
  min_child_weight = 1            # 叶节点最小样本权重
)

# 7. 训练模型
xgb_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 150,                  # 迭代次数
  watchlist = list(train = dtrain, test = dtest),
  early_stopping_rounds = 10,     # 早停轮数
  print_every_n = 10              # 打印频率
)

# 评估模型（使用confusionMatrix）
test_pred_prob <- predict(xgb_model, dtest)
test_pred <- factor(ifelse(test_pred_prob > 0.5, "yes", "no"), 
                    levels = c("no", "yes"))
test_actual <- factor(ifelse(test_data$y == 1, "yes", "no"), 
                      levels = c("no", "yes"))

confusionMatrix(test_pred, test_actual, positive = "yes")


# ROC曲线
roc_obj <- roc(test_actual, test_pred_prob)
plot(roc_obj, print.auc = TRUE)
