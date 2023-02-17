# Importing the dataset
dataset <- read.csv("Salary_Data.csv")

# Splitting the dataset into the training and test set
'install.packages("caTools")'
library(caTools)
split <- sample.split(dataset$Salary, SplitRatio = 2 / 3)
training_set <- subset(dataset, split == TRUE)
test_set <- subset(dataset, split == FALSE)

# Training the model from the training set
regressor <- lm(
  formula = Salary ~ YearsExperience,
  data = training_set
)

# Predicting the test set
y_pred <- predict(regressor, test_set)
'install.packages("ggplot2")'
library(ggplot2)
ggplot() +
  geom_point(
    aes(x = training_set$YearsExperience, y = training_set$Salary),
    colour = "red"
  ) +
  geom_line(
    aes(x = training_set$YearsExperience, y = predict(regressor, training_set)),
    colour = "blue"
  ) +
  ggtitle("Salary vs Experience (Training set)") +
  xlab("Years of Experience") +
  ylab("Salary")
