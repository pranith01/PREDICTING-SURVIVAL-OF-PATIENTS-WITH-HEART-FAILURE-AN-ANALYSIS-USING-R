
# Load required libraries
library(dplyr)       # For data manipulation
library(ggplot2)     # For data visualization
library(sampling)    # For sampling methods
library(tree)        # For building decision trees
library(caret)       # For machine learning model evaluation

# Read the dataset
heart <- read.csv("heart_failure_clinical_records_dataset.csv")

# Explore the structure, summary, and first few rows of the dataset
glimpse(heart)
summary(heart)
head(heart)

# Check the data types and missing values
glimpse(heart)
sum(is.na(heart$anaemia))
sum(is.na(heart$DEATH_EVENT))

# Convert variables to factors
heart$anaemia <- factor(heart$anaemia, labels = c("0 (False)", "1 (True)"))
heart$DEATH_EVENT <- factor(heart$DEATH_EVENT)


# Plot with ggplot
ggplot(heart, aes(x = anaemia, fill = DEATH_EVENT)) +
  geom_bar(stat = "count", position = "stack") +
  scale_x_discrete(labels  = c("0 (False)", "1 (True)")) +
  labs(x = "Anaemia", y = "Count") +
  theme_minimal(base_size = 12) +
  geom_label(stat = "count", aes(label = ..count..), position = position_stack(vjust = 0.5), size = 5)


# Plot with ggplot
ggplot(heart, aes(x = diabetes, fill = DEATH_EVENT)) +
  geom_bar(stat = "count", position = "stack") +
  scale_x_discrete(labels  = c("0 (False)", "1 (True)")) +
  labs(x = "Diabetes", y = "Count") +
  theme_minimal(base_size = 12) +
  geom_label(stat = "count", aes(label = ..count..), position = position_stack(vjust = 0.5), size = 5)


# Plot with ggplot
ggplot(heart, aes(x = high_blood_pressure, fill = DEATH_EVENT)) +
  geom_bar(stat = "count", position = "stack") +
  scale_x_discrete(labels  = c("0 (False)", "1 (True)")) +
  labs(x = "High blood pressure", y = "Count") +
  theme_minimal(base_size = 12) +
  geom_label(stat = "count", aes(label = ..count..), position = position_stack(vjust = 0.5), size = 5)


# Plot with ggplot
ggplot(heart, aes(x = sex, fill = DEATH_EVENT)) +
  geom_bar(stat = "count", position = "stack") +
  scale_x_discrete(labels  = c("0 (Female)", "1 (Male)")) +
  labs(x = "Sex", y = "Count") +
  theme_minimal(base_size = 12) +
  geom_label(stat = "count", aes(label = ..count..), position = position_stack(vjust = 0.5), size = 5)


ggplot(heart, aes(x=age, fill = DEATH_EVENT)) + 
  geom_boxplot()+ coord_flip() +theme_minimal(base_size = 12) +
  stat_boxplot(  position = "dodge2")


ggplot(heart, aes(x=creatinine_phosphokinase, fill = DEATH_EVENT)) + 
  geom_boxplot()+ coord_flip()+ theme_minimal(base_size = 12) +
  stat_boxplot(position = "dodge2")

ggplot(heart, aes(x=ejection_fraction, fill = DEATH_EVENT)) + 
  geom_boxplot()+ coord_flip() +theme_minimal(base_size = 12) +
  stat_boxplot(  position = "dodge2")

ggplot(heart, aes(x=platelets, fill = DEATH_EVENT)) + 
  geom_boxplot()+ coord_flip() + theme_minimal(base_size = 12) +
  stat_boxplot(  position = "dodge2")

ggplot(heart, aes(x=serum_creatinine, fill = DEATH_EVENT)) + 
  geom_boxplot()+ coord_flip() + theme_minimal(base_size = 12) +
  stat_boxplot(  position = "dodge2")

ggplot(heart, aes(x=serum_sodium, fill = DEATH_EVENT)) + 
  geom_boxplot()+ coord_flip() +theme_minimal(base_size = 12) +
  stat_boxplot(  position = "dodge2")

data = heart

# Splitting the data into training and testing sets
set.seed(5)
idx <- sampling:::strata(heart, stratanames = c('DEATH_EVENT'), 
                         size = c(3/4*96,3/4*203), method = 'srswor')
train <- heart[idx$ID_unit,]
test <- heart[-idx$ID_unit,]

# Logistic regression model
lr <- glm(DEATH_EVENT ~ ., family = binomial, data = train)
summary(lr)

# Making predictions and evaluating the model
preds <- predict(lr, test, type = "response")
Predict <- rep(0, dim(test)[1])
Predict[preds >= 0.5] <- 1
Actual <- test$DEATH_EVENT
table(Predict, Actual)

# Feature selection and logistic regression model
lr <- glm(DEATH_EVENT ~ age + ejection_fraction + serum_creatinine + serum_sodium + time, family = binomial,
          data = train)
summary(lr)


# Making predictions and evaluating the model
preds<-predict(lr, test, type="response")
Predict<-rep(0,dim(test)[1])
Predict[preds>=0.5]=1
Actual<-test$DEATH_EVENT
table(Predict, Actual)

# Scaling features for logistic regression
features <- subset(heart, select = c(age, ejection_fraction, serum_creatinine, serum_sodium, time))
scaled <- scale(features)
DEATH_EVENT <- heart$DEATH_EVENT
new_df <- data.frame(scaled, DEATH_EVENT)

# Splitting the scaled data into training and testing sets
idx=sample(1:nrow(new_df),3/4*nrow(new_df))
train=new_df[idx,]
test=new_df[-idx,]

# Logistic regression with scaled features
new_lr<-glm(DEATH_EVENT~age+ejection_fraction+serum_creatinine+serum_sodium+time, family=binomial, data=train)
summary(new_lr)

# Making predictions and evaluating the model
preds<-predict(new_lr, test, type="response")
Predict<-rep(0,dim(test)[1])
Predict[preds>=0.5]=1
Actual<-test$DEATH_EVENT
table(Predict, Actual)

# Classification Tree

# Splitting data for classification tree
set.seed(5)
idx <- sampling:::strata(heart, stratanames = c('DEATH_EVENT'), size = c(3/4*96,3/4*203), method = 'srswor')
train <- heart[idx$ID_unit,]
test <- heart[-idx$ID_unit,]


# Creating classification tree
tree.class<-tree(factor(DEATH_EVENT)~., train)
summary(tree.class)

# Plotting the classification tree
plot(tree.class)
text(tree.class, pretty=0)

# Making predictions and evaluating the model
tree.pred<-predict(tree.class,test,type = "class")
table(tree.pred,test$DEATH_EVENT)

# Misclassification rate
print("misclassification rate")
print(mean(tree.pred!=test$DEATH_EVENT))


# Accuracy
print("the accuracy ")
mean(tree.pred==test$DEATH_EVENT)

set.seed(3)
cv.class<-cv.tree(tree.class, FUN = prune.misclass) 
plot(cv.class$size, cv.class$dev,type="b")

prune.class=prune.misclass(tree.class,best=2)
plot(prune.class)
text(prune.class,pretty=0)

prune.pred=predict(prune.class,test,type="class")
table(prune.pred,test$DEATH_EVENT)

print("misclassification rate")

print(mean(prune.pred!=test$DEATH_EVENT))

print("the accuracy ")

mean(prune.pred==test$DEATH_EVENT)


set.seed(10)
folds<-createFolds(factor(data$DEATH_EVENT), k=10)

misclassification_tree<-function(idx){
  Train<-data[-idx,]
  Test<-data[idx,]
  fit_tree<-tree(DEATH_EVENT~., data=Train)
  pred_tree<-predict(fit_tree,Test,type = "class")
  return(1-(mean(pred_tree == Test$DEATH_EVENT)))
}

set.seed(10)
print("Classification Tree of Misclassification Rate")

mis_rate_tree=lapply(folds,misclassification_tree)
print(mean(as.numeric(mis_rate_tree)))

print("Classification Tree of the accuracy rate")

1-mean(as.numeric(mis_rate_tree))

library(caret)
set.seed(10)
folds<-createFolds(factor(data$DEATH_EVENT), k=2)

misclassification_tree<-function(idx){
  Train<-data[-idx,]
  Test<-data[idx,]
  fit_tree<-tree(DEATH_EVENT~., data=Train)
  pred_tree<-predict(fit_tree,Test,type = "class")
  return(1-(mean(pred_tree == Test$DEATH_EVENT)))
}

set.seed(10)
print("Classification Tree of Misclassification Rate")

mis_rate_tree=lapply(folds,misclassification_tree)
print(mean(as.numeric(mis_rate_tree)))

print("Classification Tree of the accuracy rate")

1-mean(as.numeric(mis_rate_tree))















