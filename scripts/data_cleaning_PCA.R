library(dplyr)
library(ggplot2)
library(corrplot)
library(car)
library(factoextra)
library(missForest)


# load data
data <- read.csv("datasets\\data_outliers_processed.csv")


# process NA
colSums(data == "unknown")
data[data == "unknown"] <- NA


# remove default
table(data$default)
data$default <- NULL


# categorical -> factor
data <- data %>%
  mutate(across(where(is.character), as.factor))


# use missForest to process NA
set.seed(123)
data_filled <- missForest(data)
data <- data_filled$ximp


# num and cat splitting
num_vars <- c("age", "campaign", "pdays", "previous", 
              "emp.var.rate", "cons.price.idx", "cons.conf.idx", 
              "euribor3m", "nr.employed")
cat_vars <- setdiff(names(data), c(num_vars, "y"))


# cat -> factor
cat_dummy <- model.matrix(~ job + marital + education + 
                            housing + loan + contact + month + 
                            day_of_week + poutcome - 1, data = data)


# merge num_vars and cat_dummy
data_processed <- cbind(data[, num_vars], cat_dummy)
head(data_processed)


# == Multicollinearity Detection ==


# correlation coefficient matrix
cor_matrix_num <- cor(data_processed[, num_vars], use = "complete.obs")
cor_matrix_num


# heat map of numeric variables
corrplot(cor_matrix_num, method = "color", 
         type = "upper", order = "hclust", 
         tl.col = "black", tl.srt = 45,
         title = "heat map of numeric variables",
         addCoef.col = "black", number.cex = 0.7)


# VIF
# convert y to num
y_numeric <- ifelse(data$y == "yes", 1, 0)
vif_data <- data_processed[, num_vars]
vif_data$y_numeric <- y_numeric


# build linear model
vif_model <- lm(y_numeric ~ ., data = vif_data)
cat("VIF of numeric variables: ", vif(vif_model), "\n")


# == PCA ==


pca_data <- data_processed


# execute PCA
pca_result <- prcomp(pca_data, center = TRUE, scale. = TRUE)
summary_pca <- summary(pca_result)
summary_pca


# scree plot
fviz_eig(pca_result, addlabels = TRUE, 
         barfill = "steelblue", barcolor = "steelblue",
         main = "scree plot of PCA")


# choose the number of principal components
cumulative_var <- summary_pca$importance["Cumulative Proportion", ]
n_components <- which(cumulative_var >= 0.8)[1]


# extract scores of chosen principal components
head(pca_result$x) # x are the scores

pca_chosen_score <- pca_result$x[, 1:n_components]
pc_scores <- as.data.frame(pca_chosen_score)

names(pc_scores) <- paste0("PC", 1:n_components) # name the columns of PC


# merge data
data_with_pc <- cbind(pc_scores, y_numeric)
names(data_with_pc)[names(data_with_pc) == "y_numeric"] <- "y"
glimpse(data_with_pc)

# save data
write.csv(data_with_pc, "datasets/data_with_pca.csv", row.names = FALSE)
