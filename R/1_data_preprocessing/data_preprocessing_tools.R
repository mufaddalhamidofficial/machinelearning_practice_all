# Importing the dataset
dataset <- read.csv("Data.csv")

# Taking care of missing data
dataset$Age <- ifelse(
  test = is.na(dataset$Age),
  yes = ave(dataset$Age, FUN = function(x) mean(x, na.rm = TRUE)),
  no = dataset$Age
)
dataset$Salary <- ifelse(
  test = is.na(dataset$Salary),
  yes = ave(dataset$Salary, FUN = function(x) mean(x, na.rm = TRUE)),
  no = dataset$Salary
)


# Encoding categorical data
dataset$Country <- factor(
  x = dataset$Country,
  levels = c("France", "Spain", "Germany"), labels = c(1, 2, 3)
)
dataset$Purchased <- factor(
  x = dataset$Purchased,
  levels = c("Yes", "No"), labels = c(0, 1)
)


# Splitting the dataset into the training and test set
# install.packages("caTools")
library(caTools)
split <- sample.split(dataset$Purchased, 0.8)
training_set <- subset(dataset, split == TRUE)
test_set <- subset(dataset, split == FALSE)

# Feature Scaling
training_set[, 2:3] <- scale(training_set[, 2:3])
test_set[, 2:3] <- scale(test_set[, 2:3])
