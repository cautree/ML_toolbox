install.packages("curl")
data(mtcars)
model = lm(mpg ~hp, mtcars[1:20,])

# predict in-sample
predicted = predict(model, mtcars[1:20,], type="response")

#calculate RMSE
actual = mtcars[1:20,"mpg"]
sqrt(mean((predicted-actual)^2))

# predict on out of sample
predicted_out = predict(model, mtcars[21:32,], type="response")
actual_test = mtcars[21:32,"mpg"]
sqrt(mean((predicted_out-actual_test)^2))

##################################
#diamonds dataset
#################################
#diamonds dataset
# Fit lm model: model
model = lm(price ~., data=diamonds)

#names(diamonds)

# Predict on full data: p
p = predict(model, diamonds[, -7], type="response")
#p = predict(model, diamonds[, -7])

# Compute errors: error
actual = diamonds$price
error = p-actual

# Calculate RMSE
sqrt(mean(error^2))

# Set seed
set.seed(42)

# Shuffle row indices: rows
rows=sample(nrow(diamonds))

# Randomly order data
diamonds=diamonds[rows,]
# Determine row to split on: split
split = round(nrow(diamonds)*0.8)

# Create train
train = diamonds[1:split,]

# Create test
test = diamonds[(split+1):nrow(diamonds),]
# Fit lm model on train: model
model = lm(price ~., data=train)


# Predict on test: p
p = predict(model, test[,-7])

# Compute errors: error
error = p- test$price

# Calculate RMSE
sqrt(mean(error^2))

# cross validation on the diamond dataset
# Fit lm model using 10-fold CV: model
model <- train(
  price~., diamonds,
  method = "lm",
  trControl = trainControl(
    method = "cv", number =10,
    verboseIter = TRUE
  )
)

# Print model to console
model


##############################
#Boston data set, repeated cross validation
#############################
# Fit lm model using 5 x 5-fold CV: model
library(MASS)
data("Boston")
names(Boston)
model <- train(
  medv ~ ., Boston,
  method = "lm",
  trControl = trainControl(
    method = "cv", number = 5,
    repeats = 5, verboseIter = TRUE
  )
)

# Print model to console
model
# Predict on full Boston dataset
dim(Boston)
predict(model, Boston[,-14])


#########################################
#cross validation on the mtcars dataset
###########################################
library(caret)
set.seed(42)


#caret supports many types of cross-validation, and you can specify which type of cross-validation
#and the number #of cross-validation folds with the trainControl() function, 
#which you pass to the trControl argument in train():
#fit linear regression model
model = train(mpg ~hp, mtcars,
              method="lm",
              trControl =trainControl(
                method="cv", number=10,
                verboseIter = TRUE  # gives the progress log
              ))

##############################################
# classification problem  sonar dataset
###################################################
#install.packages("mlbench")
library(mlbench)
data("Sonar")

# Shuffle row indices: rows
rows = sample(nrow(Sonar))

# Randomly order data: Sonar
Sonar = Sonar[rows,]

# Identify row to split on: split
split <- round(nrow(Sonar) * 0.6)

# Create train
train = Sonar[1:split,]

# Create test
test = Sonar[(split+1):nrow(Sonar),]

#glm() is a more advanced version of lm() that allows for more varied types of regression models, 
#aside from plain vanilla ordinary least squares regression. 

#Be sure to pass the argument family = "binomial" to glm() to 
#specify that you want to do logistic (rather than linear) regression. 
#glm(Target ~ ., family = "binomial", dataset)

# Fit glm model: model
model = glm(Class~.,family = "binomial",data=train)
# Predict on test: p
p = predict(model, test, type="response")
summary(p)

#fit a model
names(Sonar)
model = glm(Class~., family = binomial(link="logit"), train)
p=predict(model,test[,-1], type="response")  # type = response can not be omitted
summary(p)

# turn prob into classes and look at their frequecies
p_class = ifelse(p>0.5, "M","R")
table(p_class)

table(p_class, test[["Class"]])

# use caret's helper function to calculate additional statistics
confusionMatrix(p_class, test[["Class"]])

#Predicted classes are based off of predicted probabilities plus a classification threshold.
# Apply threshold of 0.9: p_class
p_class =ifelse(p >0.9, "M","R")

# Create confusion matrix
confusionMatrix(p_class,test$Class)


#####################################################
#ROC curves at every possible threshold 
######################################################

# let the computer to try out different threshold, and calculate the true positive rate

#create ROC curve
library(caTools)
colAUC(p, test[["Class"]], plotROC=TRUE)
#X-axis: false positive rate
#Y-axis: true positive rate
# each point represents a confusion matrix we do not have to analyze by hand

#an ROC curve is a really useful shortcut for summarizing the performance of a classifier 
#over all possible thresholds. 
#This saves you a lot of tedious work computing class predictions for many different thresholds 
#and examining the confusion matrix for each.

#My favorite package for computing ROC curves is caTools, which contains a function called colAUC(). 
#This function is very user-friendly and can actually calculate ROC curves for multiple predictors at once. 
#In this case, you only need to calculate the ROC curve for one predictor, e.g.:
  
#colAUC(predicted_probabilities, actual, plotROC = TRUE)
#The function will return a score called AUC (more on that later) and the plotROC= TRUE argument 
#will return the plot of the ROC curve for visual inspection.

#############################################
#from ROC to AUC
#############################################
#area under the curve
# model that do random prediction tends to fall along the diagonal line
# 100% correct model is a box, the area under the curve is exactly 1
# the area under the curve for a random model is 0.5

# model accuracy: AUC, area under the curve

#defining AUC
#ranges from 0 to 1
#0.5 random guessing
# 1 model always right

#ROC curve is a very useful, single-number summary of a model's ability to discriminate 
#the positive from the negative class (e.g. mines from rocks). 
#An AUC of 0.5 is no better than random guessing, an AUC of 1.0 is a perfectly predictive model, 
#and an AUC of 0.0 is perfectly anti-predictive (which rarely happens).

#This is often a much more useful metric than simply ranking models by their accuracy at a set threshold, 
#as different models might require different calibration steps (looking at a confusion matrix at each step) to 
#find the optimal classification threshold for that model.

##########??????????????????????????????????????
#You can use the trainControl() function in caret to use AUC (instead of acccuracy), 
#to tune the parameters of your models. The twoClassSummary() convenience function allows you to do this easily.

#When using twoClassSummary(), be sure to always include the argument classProbs = TRUE or 
#your model will throw an error! (You cannot calculate AUC with just class predictions. 
#You need to have class probabilities as well.)

# Create trainControl object: myControl
myControl <- trainControl(
  method = "cv",
  number = 10,
  summaryFunction = twoClassSummary,
  classProbs = TRUE, # IMPORTANT!
  verboseIter = TRUE
)


#################################################
# Using custom train Control
####################################################
#Now that you have a custom trainControl object, it's easy to fit caret models that use AUC rather than 
#accuracy to tune and evaluate the model. 
#You can just pass your custom trainControl object to the train() function via the trControl argument, e.g.:

#train(<standard arguments here>, trControl = myControl)
#This syntax gives you a convenient way to store a lot of custom modeling parameters 
#and then use them across multiple different calls to train().

# Train glm with custom trainControl: model
model=train(Class~., data=Sonar, method="glm", trControl = myControl)

#############################
#Random Forest Models
############################

#random forest models are much more flexible than linear models, 
#and can model complicated nonlinear effects as well as automatically capture interactions between variables. 
#They tend to give very good results on real world data, 
#so let's try one out on the wine quality dataset, where the goal is to predict the human-evaluated quality of a batch of wine, 
#given some of the machine-measured chemical and physical properties of that batch.

#Fitting a random forest model is exactly the same as fitting a generalized linear regression model, as you did in the previous chapter. 
#You simply change the method argument in the train function to be "ranger". 
#The ranger package is a rewrite of R's classic randomForest package and fits models much faster, 
#but gives almost exactly the same results. 
#We suggest that all beginners use the ranger package for random forest modeling.

library(caret)
library(mlbench)
data(Sonar)
set.seed(42)
model = train(Class~.,
              data=Sonar,
              method="ranger",
              tuneLength =10)
# plot the results for different value of mtry
# what is mtry
plot(model)


####################################################
#fit a random forest data on wine dataset
#####################################################

# Fit random forest: model
#names(wine)
model <- train(
  quality~.,
  tuneLength =1,
  data = wine, method = "ranger",
  trControl = trainControl(method = "cv", number = 5, verboseIter = TRUE)
)

##########################
#try a longer tune length
###########################
#random forest models have a primary tuning parameter of mtry, 
#which controls how many variables are exposed to the splitting search routine at each split. 
#For example, suppose that a tree has a total of 10 splits and mtry = 2. 
#This means that there are 10 samples of 2 predictors each time a split is evaluated.

# Fit random forest: model
model <- train(
  quality~.,
  tuneLength = 3,
  data = wine, method = "ranger",
  trControl = trainControl(method = "cv", number = 5, verboseIter = TRUE)
)

# Print model to console
model

# Plot model
plot(model)


############################
#custom tuning example
############################
#define a custom tuning grid
myGrid = data.frame(mtry=c(2,3,4,5,10,20))

#fit the model with a custom turning grid
set.seed(42)
model = train(Class~.,
              data=Sonar,
              method="ranger",
              tuneGrid = myGrid)
plot(model)


# Fit random forest: model
names(wine)

model <- train(
  quality~.,
  tuneGrid = data.frame(mtry=c(2,3,7)),
  data = wine, method = "ranger",
  trControl = trainControl(method = "cv", number = 5, verboseIter = TRUE)
)

# Print model to console
model
# Plot model
plot(model)

#########################################
#introducing glmnet
########################################
# extension of glm models with built-in variable selection helps deal with collinearity 
# and small samples sizes
# Two primary forms
# lasso regression : penalize number of non-zero coneffients
# ridge regression: penalize absolute magnitude of coeffienits
# Attemps to find a parsimonious(ie simple) model

###################################
#tuning glmnet models
###################################
#combination of lasso and ridge regression can fit a mix of the two models
#alpha[0,1]: pure lasso to pure ridge
#lambda(0, infinity): size of the penalty

# example "dont overfit"
overfit = read.csv("http://s3.amazonaws.com/assets.datacamp.com/production/course_1048/datasets/overfit.csv")

#make a custom trainControl
myControl = trainControl(
  method="cv", number = 10,
  summaryFunction = twoClassSummary,
  classProbs = TRUE, # super important
  verboseIter = TRUE
)

# try the defaults
set.seed(42)
model= train(y~., overfit,
             method="glmnet",
             trControl=myControl)
plot(model)

#What's the advantage of glmnet over regular glm models?
#glmnet models place constraints on your coefficients, which helps prevent overfitting.

#Classification problems are a little more complicated than regression problems 
#because you have to provide a custom summaryFunction to the train() function 
#to use the AUC metric to rank your models. 
#Start by making a custom trainControl, as you did in the previous chapter. 
#Be sure to set classProbs = TRUE, otherwise the twoClassSummary for 
#summaryFunction will break.

# Create custom trainControl: myControl
myControl <- trainControl(
  method = "cv", number = 10,
  summaryFunction = twoClassSummary,
  classProbs = TRUE, # IMPORTANT!
  verboseIter = TRUE
)

#glmnet is an extention of the generalized linear regression model (or glm) that 
#places constraints on the magnitude of the coefficients to prevent overfitting. 
#This is more commonly known as "penalized" regression modeling and is a very useful technique on datasets 
#with many predictors and few values.

#glmnet is capable of fitting two different kinds of penalized models, controlled by the alpha parameter:

#Ridge regression (or alpha = 0)
#Lasso regression (or alpha = 1)
# Fit glmnet model: model

head(overfit)
model <- train(
  y~., overfit,
  method = "glmnet",
  trControl = myControl
)

# Print model to console
model
model$results
# Print maximum ROC statistic
max(model$results$ROC)

##########################################
# glmnet turning model
##########################################
myGrid = expand.grid(
  alpha=0:1,
  lambda =seq(0.0001,0.1,length=10)
)

#fit a model
set.seed(42)
model = train(y~., overfit, method="glmnet",
              tuneGrid = myGrid, trControl=myControl)

plot(model)


plot(model$finalModel)


#Why use a custom tuning grid for a glmnet model?
#The default tuning grid is very small and there are many 
#more potential glmnet models you want to explore.


#the glmnet model actually fits many models at once (one of the great things about the package). 
#You can exploit this by passing a large number of lambda values, which control the amount of penalization in the model. 
#train() is smart enough to only fit one model per alpha value and pass all of the lambda values at once for simultaneous fitting.

#My favorite tuning grid for glmnet models is:
expand.grid(alpha = 0:1,
            lambda = seq(0.0001, 1, length = 100))

#This grid explores a large number of lambda values (100, in fact), from a very small one to a very large one. 
#(You could increase the maximum lambda to 10, but in this exercise 1 is a good upper bound.)
  
#You also look at the two forms of penalized models with this tuneGrid: 
#ridge regression and lasso regression. alpha = 0 is pure ridge regression, 
#and alpha = 1 is pure lasso regression. You can fit a mixture of the two models 
#(i.e. an elastic net) using an alpha between 0 and 1. For example, alpha = .05 would be 95% 
#ridge regression and 5% lasso regression.

# Train glmnet with custom trainControl and tuning: model
myGrid = expand.grid(
  alpha=0:1,
  lambda =seq(0.0001,1,length=20)
)

model <- train(
  y~., overfit,
  tuneGrid = myGrid,
  method = "glmnet",
  trControl = myControl
)


# Print maximum ROC statistic
max(model[["results"]][["ROC"]])

####################################
#manipulate your data
####################################
data(mtcars)
names(mtcars)
set.seed(42)
# fake a NA data
mtcars[sample(1:nrow(mtcars),10),"hp"] =NA
summary(mtcars$hp)
Y= mtcars$mpg
X = mtcars[,2:4]
library(caret)
model = train(X,Y)

# the model is not working now, we can impute the missing value using median
model = train(X,Y, preProcess="medianImpute")
model

# with missing value, 
#train() function in caret contains an argument called preProcess, 
#which allows you to specify that median imputation should be used to fill in the missing values

###############################
#breast cancer dataset
###############################
# Apply median imputation: model
data("BreastCancer")
dim(BreastCancer)
names(BreastCancer)
summary(BreastCancer)
breast_cancer_x = BreastCancer[,c(2:(ncol(BreastCancer)-1))]
breast_cancer_y= BreastCancer[,ncol(BreastCancer)]
breast_cancer_y

model <- train(
  x = breast_cancer_x, y =breast_cancer_y,
  method = "glm",
  trControl = myControl,
  preProcess = "medianImpute"
)

model

##############################
#KNN impuation: dealing with missing values
############################
# median impuation is fast, but...
#can produce incorrect results if data missing not at random
# KNN impuation imputes based on "similar "  non-missing rows, 
# KNN impuation can deal with missing not at random problem

###################################
#predend smaller cars dont report horsepower
#median imputation is incorrect in this case
# as it is missing not at random

# generate data with missing values
data(mtcars)
mtcars[mtcars$disp<140,"hp"]=NA
Y=mtcars$mpg
X=mtcars[,2:4]


#use median imputation
set.seed(42)
model=train(X,Y,method="glm",
            preProcess="medianImpute")
model
min(model$results$RMSE)


##use KNN imputation
# somehow did not work as in the lectures, not sure why
set.seed(42)
model= train(X,Y, method="glm", preProcess="knnImpute")
min(model$results$RMSE)


###################################
#the wide world of preProcess
#####################################
#you can do a lot more than median or knn imputation
#can chain together multiple preprocessing steps
# common "recipe" for linear models (order matters)
#median impuation > center> scale> fit glm
# by the way, PCA always happen after center and scaling

################################
# chained preprocess using mtcars as an example
#################################
data(mtcars)
set.seed(42)
mtcars[sample(1:nrow(mtcars),10),"hp"] =NA
Y=mtcars$mpg
X=mtcars[,2:4]

#use linear model "recipe"
set.seed(42)
model = train(X,Y, method="glm",
              preProcess=c("medianImpute","center","scale"))

min(model$results$RMSE)


# add PCA in the preprocess
set.seed(42)
model = train(X,Y, method="glm",
              preProcess=c("medianImpute","center","scale","pca"))
min(model$results$RMSE)

######################
#preprocesing cheat sheet
#####################
# start with median imputation, try KNN impuation, if your data is missing not at random
# for linear models..
# always center and scale
# try PCA and spatial sign transformation
# tree based models dont need much preprocessing


########################
#tips: remove extremely low variance variables prior modeling
#######################
mtcars[sample(1:nrow(mtcars),10),"hp"] =NA
Y=mtcars$mpg
X=mtcars[,2:4]
X$bad =1  # add this constant, the model as below will not work
model = train(X,Y, method="glm", preProcess =c("medianImpute","center","scale","PCA"))

# however, caret "zv"  or "nzv" can solve this problem
model = train(X,Y, method="glm",
              preProcess = c("zv","medianImpute","center","scale"))
min(model$results$RMSE)


#caret contains a utility function called nearZeroVar() for removing low variance variables 
#to save time during modeling.

# precessing method comparasion
# "zv" < "nzv" < "zv"+"PCA"  # as PCA kepted nearly zero variance variable, but thouse variables are sueful

#nearZeroVar() takes in data x, then looks at the ratio of the most common value 
#to the second most common value, freqCut, and the percentage of distinct values 
#out of the number of total samples, uniqueCut. 
#By default, caret uses freqCut = 19 and uniqueCut = 10, 
#which is fairly conservative. 
#I like to be a little more aggressive and use freqCut = 2 and uniqueCut = 20 
#when calling nearZeroVar().
# Identify near zero variance predictors: remove_cols
remove_cols <- nearZeroVar(bloodbrain_x, names = TRUE, 
                           freqCut = 2, uniqueCut = 20)
# Get all column names from bloodbrain_x: all_cols
all_cols=colnames(bloodbrain_x)

# Remove from data: bloodbrain_x_small
bloodbrain_x_small <- bloodbrain_x[ , setdiff(all_cols, remove_cols)]


names(mtcars)[nearZeroVar(mtcars)]


##############################################
# using PCA as an alternative to nearZeroVar()
################################################
#An alternative to removing low-variance predictors is to run PCA on your dataset. 
#This is sometimes preferable because it does not throw out all of your data: 
#many different low variance predictors may end up combined into one high variance PCA variable, which might have a positive impact on your model's accuracy.

#This is an especially good trick for linear models: the pca option in the preProcess argument 
#will center and scale your data, combine low variance variables, 
#and ensure that all of your predictors are orthogonal. 
#This creates an ideal dataset for linear regression modeling, and can often improve the accuracy of your models.


#Note that the PCA model's accuracy is slightly higher than the nearZeroVar() model from the previous exercise. 
#PCA is generally a better method for handling low-information predictors than throwing them out entirely.

###########################
#predictive modeling on real life data
###########################
library(caret)
library(C50)
data("churn")
table(churnTrain$churn)/nrow(churnTrain)

set.seed(42)
myFolds = createFolds(churnTrain$churn,k=5)

# compare class distribution
i=myFolds$Fold1
table(churnTrain$churn[i])/length(i)

#use folds to create a trainControl object
myControl = trainControl(
  summaryFunction = twoClassSummary,
  classProbs = TRUE,
  verboseIter = TRUE,
  savePredictions = TRUE,
  index=myFolds
)


#############################
#Why reuse a trainControl?
###############################
#So you can use the same summaryFunction and tuning parameters for multiple models.
#So you don't have to repeat code when fitting multiple models.
#So you can compare models on the exact same training and test data.

###############################
#glmnet review
# linear model with built-in variable selection
#great baseline model
# advantages
#fits quickly
#ignores noisy variables
#provides interpretable coeffieinents


myGrid = expand.grid(
  alpha=0:1,
  lambda =seq(0.0001,1,length=20)
)
#fit the model

set.seed(42)
model_glmnet = train(
  churn~., churnTrain,
  metric = "ROC",
  method="glmnet",
  tuneGrid = myGrid,
  trControl = myControl
)

plot(model_glmnet)

# plot the coeffecients
plot(model_glmnet$finalModel)



model_glmnet = train(
  churn~., churnTrain,
  metric="ROC",
  method ="glmnet",
  trControl=myControl
)


####################################
#reintroduce random forest
#####################################
# slower to fit than glmnet
# less interpretable
# often(but not always) more accurate than glmnet
# Easier to tune
# require litte preprocessing
# capture threhold effects and variable interations

#Random forst on churn data
set.seed(42)
model_rf = train(
  churn~., churnTrain,
  metric= "ROC",
  method = "ranger",
  trControl= myControl
)
plot(model_rf)


###########################
#if you do the following, basically they are the same
model_rf = train(
  x=churn_x,
  y=churn_y,
  metric="ROC",
  method ="ranger",
  trControl=myControl
)



############################################
# comparing models
#make sure they were fit on the same data!
#selection criteria
## highest average AUC
## lowest standard deviation in AUC
# The resamples() function is your friend

#You can compare models in caret using the resamples() function, 
#provided they have the same training data and use the same trainControl object 
#with preset cross-validation folds. resamples() takes as input a list of models 
#and can be used to compare dozens of models at once 
#(though in this case you are only comparing two models).



# makes a list
model_list = list(
  glmnet = model_glmnet,
  rf=model_rf
)

# collect resamples from the CV folds
resamps= resamples(model_list)
resamps
summary(resamps)


###################################################
#more on resample
####################################################
#comparing models
## resamples has tons of cool methods
## one of the  author's favorite functions
## inspired the caretEnsemble package

bwplot (resamps, metric="ROC")  # box and whisker plot
dotplot(resamps, metric="ROC")  # dot plot, visual simipler
densityplot(resamps, metric="ROC")  # kernal density 
xyplot(resamps, metric="ROC")  # scatter plot

###if you have a lot of models for comparision
#using dotplot
dotplot(lots_of_models, metric="ROC")



#caret provides a variety of methods to use for comparing models. 
#All of these methods are based on the resamples() function.
#My favorite is the box-and-whisker plot, 
#which allows you to compare the distribution of predictive accuracy (in this case AUC) 
#for the two models.

#In general, you want the model with the higher median AUC, 
#as well as a smaller range between min and max AUC.

#You can make this plot using the bwplot() function, 
#which makes a box and whisker plot of the model's out of sample scores. 
#Box and whisker plots show the median of each distribution as a line 
#and the interquartile range of each distribution as a box around the median line. 
#You can pass the metric = "ROC" argument to the bwplot() function to 
#show a plot of the model's out-of-sample ROC scores and choose the model 
#with the highest median ROC.
