#####
# AUTHOR: Vamsi Krishna Muppala
# FILENAME: Assignment1
# SPECIFICATION: To prepare a dataset for analysis, to build and evaluate a model using logistic regression and compare the accuracies and find the best model amount the two different dataset.After finding the higest accuracy model, compute the ROC,PR curve and k-folds for the dataset. 
# FOR: CS 5331 Machine Learning and Information Security Section 001
#####

#installing the packages of ggplot and ggally:
install.packages("ggplot2")
library(ggplot2)
install.packages("GGally")
library(GGally)

#reading the dataset:
#varaible ds means dataset
ds <- read.csv("datafile.csv")
summary(ds)

#pairs plot for a1 to a4 and a5 to a8:
pairs(ds[ , 1:4], main = "pair plot for a1 to a4")
pairs(ds[ , 5:8], main = "pair plot for a5 to a8")

#ggpairs() plot for a1 to a4 and a5 to a8:
ggpairs(ds,  mapping = NULL,  columns = 1:4,  title = "ggapir plot for a1 to a4")
ggpairs(ds,  mapping = NULL,  columns = 5:8,  title = "ggpair plot for a5 to a8")

#pairs: The diagonal has the names of the variables like a1,a2,a3,a4,a5,a6,a7,a8.The other cells of the plot show a scatterplot (i.e. correlation plot) of each variable combination within each other of our data frame.
#ggpairs: The diagonal consists of the densities of the  variables and the upper portion consist of the correlation coefficients between the variables and the lower portion consists of the scattered plot between the variables.

#to find number of 1s and 0s in a8:
numo <- 0 #num0 = varaiable that is used to find the number of ones in column a8 for the dataset ds
numz <- 0 #num1 = varaiable that is used to find the number of zeros in column a8 for the dataset ds
for (i in ds$a8) {
  if(i == 0)  numz = numz+1
  else numo = numo+1
}
#Imbalance: The number of zeroes is 326 and number of ones is 174, therefore imbalance is 152

#create a random dataset with 150 zeros and 150 ones from column a8 of dataset ds:
#set.seed <- makes sure get same results for randomization: 
set.seed(2508)
#set of random rows of number of 0's and 1's of column a8:
ss_zeros= ds[sample(which(ds$a8 == "0" ),150),] #ss_zeros<- variable of set that contains random number of 150 zeros from the column a8 of dataset ds
ss_ones= ds[sample(which( ds$a8 == "1" ),150),] #ss_ones<- variable of set that contains random number of 150 ones from the column a8 of dataset ds
#the sample dataset:
ss = rbind(ss_zeros,ss_ones) #ss<- datasetof total 300 columns that conatins random of 150 zeros and 150 ones from column a8 of dataset ds

#spliting sample set into training and testing sets:
#70:30 
data = sample(nrow(ss), nrow(ss)*.7) #data<- set of variable that contains the indexis of 70% of rows for column a8 of the dataset ss
train_ss <-ss[data,] #train_ss<- taining dataset to train the model that contains 70% of data from the dataset ss
test_ss <-ss[-data,] #test_ss<- testing dataset to test the model that conatins the remainings 30% of data from the dataset ss

#60:40
data1= sample(nrow(ss), nrow(ss)*.6) #data<- set of variable that contains the indexis of 60% of rows for column a8 of the dataset ss
train_ss_60 <-ss[data1,] #train_ss<- taining dataset to train the model that contains 60% of data from the dataset ss
test_ss_40 <-ss[-data1,] #test_ss<- testing dataset to test the model that conatins the remainings 30% of data from the dataset ss

##Build a Logistic Regression Model on the training data:
#binomial regression model :

#Case1:
#when the dataset ss is divided into 70percent into training(train_ss) and 30% into testing(test_ss) : 

#we are training 5 models t_1,t_2,t_3,t_4,t_5 to build model using glm with the training dataset train_ss
t_1 = glm(formula = a8 ~ a1 + a2 + a3 + a4 + a5 + a6 + a7, data = train_ss , family = binomial) 
t_2 = glm(formula = a8 ~ a1 + a3 + a5 + a7 , data = train_ss,family = binomial)
t_3 = glm(formula = a8 ~ a2 + a4 + a6 , data = train_ss,family = binomial)
t_4 = glm(formula = a8 ~ a2 + a3 + a5 , data = train_ss,family = binomial)
t_5 = glm(formula = a8 ~ a7 + a2 + a5 , data = train_ss,family = binomial)

#Use the test dataset:test_ss on the 5 models respectively to predict the probability 
test_ss$prob_1 = predict(t_1,newdata = test_ss,type = "response")
test_ss$prob_2 = predict(t_2,newdata = test_ss,type = "response")
test_ss$prob_3 = predict(t_3,newdata = test_ss,type = "response")
test_ss$prob_4 = predict(t_4,newdata = test_ss,type = "response")
test_ss$prob_5 = predict(t_5,newdata = test_ss,type = "response")

#create the column in test dataset test_ss to contain 1 or 0 based upon test$prob for the 5 models respectively
test_ss$check_1 = as.factor(ifelse(test_ss$prob_1 > 0.5, 1, 0))
test_ss$check_2 = as.factor(ifelse(test_ss$prob_2 > 0.5, 1, 0))
test_ss$check_3 = as.factor(ifelse(test_ss$prob_3 > 0.5, 1, 0))
test_ss$check_4 = as.factor(ifelse(test_ss$prob_4 > 0.5, 1, 0))
test_ss$check_5 = as.factor(ifelse(test_ss$prob_5 > 0.5, 1, 0))

#make a confusion table to see how well the classifier predicted the test set and making the rows and columns start from (1,1) for all the 5 models:
confusion_matrix_1 = table(test_ss$a8,test_ss$check_1)[2:1,2:1]
confusion_matrix_2 = table(test_ss$a8,test_ss$check_2)[2:1,2:1]
confusion_matrix_3 = table(test_ss$a8,test_ss$check_3)[2:1,2:1]
confusion_matrix_4 = table(test_ss$a8,test_ss$check_4)[2:1,2:1]
confusion_matrix_5 = table(test_ss$a8,test_ss$check_5)[2:1,2:1]

#set and print accuracy
#set accuracy to (TP+TN)/(TP+FP+FN+TN), for all the 5 models
accuracy_1 = (confusion_matrix_1[1,1]+confusion_matrix_1[2,2])/(confusion_matrix_1[1,1]+confusion_matrix_1[1,2]+confusion_matrix_1[2,1]+confusion_matrix_1[2,2])
accuracy_2 = (confusion_matrix_2[1,1]+confusion_matrix_2[2,2])/(confusion_matrix_2[1,1]+confusion_matrix_2[1,2]+confusion_matrix_2[2,1]+confusion_matrix_2[2,2])
accuracy_3 = (confusion_matrix_3[1,1]+confusion_matrix_3[2,2])/(confusion_matrix_3[1,1]+confusion_matrix_3[1,2]+confusion_matrix_3[2,1]+confusion_matrix_3[2,2])
accuracy_4 = (confusion_matrix_4[1,1]+confusion_matrix_4[2,2])/(confusion_matrix_4[1,1]+confusion_matrix_4[1,2]+confusion_matrix_4[2,1]+confusion_matrix_4[2,2])
accuracy_5 = (confusion_matrix_5[1,1]+confusion_matrix_5[2,2])/(confusion_matrix_5[1,1]+confusion_matrix_5[1,2]+confusion_matrix_5[2,1]+confusion_matrix_5[2,2])

print(paste("accuracy of first set:  ",accuracy_1))
print(paste("accuracy of second set: ",accuracy_2))
print(paste("accuracy of third set:  ",accuracy_3))
print(paste("accuracy of fourth set: ",accuracy_4))
print(paste("accuracy of fifth set:  ",accuracy_5))

#case2:
#when the dataset ss is divided into 60percent into training(train_ss_60) and 40% into testing(test_ss_40): 

#we are training 5 models t_1_1,t_2_1,t_3_1,t_4_1,t_5_1 to build model using glm  with the training dataset train_ss
t_1_1 = glm(formula = a8 ~ a1 + a2 + a3 + a4 + a5 + a6 + a7, data = train_ss_60 , family = binomial)
t_2_1 = glm(formula = a8 ~ a1 + a3 + a5 + a7 , data = train_ss_60,family = binomial)
t_3_1 = glm(formula = a8 ~ a2 + a4 + a6 , data = train_ss_60,family = binomial)
t_4_1 = glm(formula = a8 ~ a2 + a3 + a5 , data = train_ss_60,family = binomial)
t_5_1 = glm(formula = a8 ~ a7 + a2 + a5 , data = train_ss_60,family = binomial)

#Use the test dataset:test_ss_40 on the 5 models respectively to predict the probability  
test_ss_40$prob_1_1 = predict(t_1_1,newdata = test_ss_40,type = "response")
test_ss_40$prob_2_1 = predict(t_2_1,newdata = test_ss_40,type = "response")
test_ss_40$prob_3_1 = predict(t_3_1,newdata = test_ss_40,type = "response")
test_ss_40$prob_4_1 = predict(t_4_1,newdata = test_ss_40,type = "response")
test_ss_40$prob_5_1 = predict(t_5_1,newdata = test_ss_40,type = "response")

#create the column in test dataset test_ss_40 to contain 1 or 0 based upon test$prob for the 5 models respectively
test_ss_40$check_1_1 = as.factor(ifelse(test_ss_40$prob_1_1 > 0.5, 1, 0))
test_ss_40$check_2_1 = as.factor(ifelse(test_ss_40$prob_2_1 > 0.5, 1, 0))
test_ss_40$check_3_1 = as.factor(ifelse(test_ss_40$prob_3_1 > 0.5, 1, 0))
test_ss_40$check_4_1 = as.factor(ifelse(test_ss_40$prob_4_1 > 0.5, 1, 0))
test_ss_40$check_5_1 = as.factor(ifelse(test_ss_40$prob_5_1 > 0.5, 1, 0))

#make a confusion table to see how well the classifier predicted the test set and making the rows and columns start from (1,1) for all the 5 models:
confusion_matrix_1_1 = table(test_ss_40$a8,test_ss_40$check_1_1)[2:1,2:1]
confusion_matrix_2_1 = table(test_ss_40$a8,test_ss_40$check_2_1)[2:1,2:1]
confusion_matrix_3_1 = table(test_ss_40$a8,test_ss_40$check_3_1)[2:1,2:1]
confusion_matrix_4_1 = table(test_ss_40$a8,test_ss_40$check_4_1)[2:1,2:1]
confusion_matrix_5_1 = table(test_ss_40$a8,test_ss_40$check_5_1)[2:1,2:1]

#set and print accuracy
#set accuracy to (TP+TN)/(TP+FP+FN+TN), for all the 5 models
accuracy_1_1 = (confusion_matrix_1_1[1,1]+confusion_matrix_1_1[2,2])/(confusion_matrix_1_1[1,1]+confusion_matrix_1_1[1,2]+confusion_matrix_1_1[2,1]+confusion_matrix_1_1[2,2])
accuracy_2_1 = (confusion_matrix_2_1[1,1]+confusion_matrix_2_1[2,2])/(confusion_matrix_2_1[1,1]+confusion_matrix_2_1[1,2]+confusion_matrix_2_1[2,1]+confusion_matrix_2_1[2,2])
accuracy_3_1 = (confusion_matrix_3_1[1,1]+confusion_matrix_3_1[2,2])/(confusion_matrix_3_1[1,1]+confusion_matrix_3_1[1,2]+confusion_matrix_3_1[2,1]+confusion_matrix_3_1[2,2])
accuracy_4_1 = (confusion_matrix_4_1[1,1]+confusion_matrix_4_1[2,2])/(confusion_matrix_4_1[1,1]+confusion_matrix_4_1[1,2]+confusion_matrix_4_1[2,1]+confusion_matrix_4_1[2,2])
accuracy_5_1 = (confusion_matrix_5_1[1,1]+confusion_matrix_5_1[2,2])/(confusion_matrix_5_1[1,1]+confusion_matrix_5_1[1,2]+confusion_matrix_5_1[2,1]+confusion_matrix_5_1[2,2])

print(paste("accuracy of first set:  ",accuracy_1_1))
print(paste("accuracy of second set: ",accuracy_2_1))
print(paste("accuracy of third set:  ",accuracy_3_1))
print(paste("accuracy of fourth set: ",accuracy_4_1))
print(paste("accuracy of fifth set:  ",accuracy_5_1))

#By comapring the accuracy of the total 10 models(5 with 70-30percent of dataset and 5 with 60-40percent of dataset):
#Highest accuracy is for t_2: lets test it using the full dataset and find the accurancy:
#Use the full dataset: ss to predict the probability
ss$prob = predict(t_2,newdata = ss,type = "response")
#create the column in full dataset:ss to contain 1 or 0 based upon test$prob for the 5 models respectively
ss$check = as.factor(ifelse(ss$prob > 0.5, 1, 0))
#make a confusion table to see how well the classifier predicted the full dataset of the model t_2 and making the rows and columns start from (1,1) for all the 5 models:
confusion_matrix_1_hacc = table(ss$a8,ss$check)[2:1,2:1]
#set and print accuracy
#set accuracy to (TP+TN)/(TP+FP+FN+TN)
accuracy_1_hacc = (confusion_matrix_1_hacc[1,1]+confusion_matrix_1_hacc[2,2])/(confusion_matrix_1_hacc[1,1]+confusion_matrix_1_hacc[1,2]+confusion_matrix_1_hacc[2,1]+confusion_matrix_1_hacc[2,2])
print(paste("accuracy of highest accu set:  ",accuracy_1_hacc))

#ROC:
#Receiver Operating Characteristic (ROC) Curve on 
#installing the packages of ROCR
install.packages("ROCR")  #perform once on your system
library("ROCR")
#From onwards we train using full dataset:
pred_ss = prediction(ss$prob,ss$a8)
#look at the accuracy of the prediction
perf_ss = performance(pred_ss, "acc")
#plot the accuracy performance
plot(perf_ss)
#plot the ROC curve using tpr versus fpr
roc_ss = performance(pred_ss,"tpr","fpr")
plot (roc_ss, colorize=T,lwd=2)
abline(a=0,b=1)
#calculate the area under the curve
auc_ss = performance(pred_ss,measure="auc")
print(auc_ss@y.values)

#ROC plot means:
#From the graph we could see the true positive is increasing and no curve in the false positive region,the area under the curve which is the accuracy is 0.81244 which is shows the true positive rate is high.

#PR curve:
#plotting the precision recall curve
pr_rec_ss = performance(pred_ss,"prec","rec")
plot(pr_rec_ss, avg= "threshold", colorize=TRUE, lwd= 3, main= "... Precision/Recall graphs ...")
#include a dotted grey line
plot(pr_rec_ss, lty=3, col="grey78", add=TRUE)
#calculate and print the area under the curve
aucpr_ss = performance(pred_ss,measure="aucpr")
print(aucpr_ss@y.values)

#PR curve means:
#the area under the curve is 0.8125 which is better as we could see the positive set belong to the match set is high and also the match cases are classified almost correctly.

#K-fold cross-validation
#to be able to run several different test sets, use the n-fold cross-validation technique
#set the number of folds
nfolds = 9
#Randomly shuffle the data
dataset_shuffle<-ss[sample(nrow(ss)),]
#Create folds (makes a vector marking each fold for 9 folds)
folds <- cut(seq(1,nrow(dataset_shuffle)),breaks=nfolds,labels=FALSE)
average_accuracy = 0.0
for(i in 1:nfolds){
  #Choose 1 fold and separate the testing and training data 
  testIndexes <- which(folds==i,arr.ind=TRUE)
  testData <- dataset_shuffle[testIndexes, ]
  trainData <- dataset_shuffle[-testIndexes, ]
  #Build the model using glm of logistic regression with the highest accuraccy model
  t_mod = glm(formula = a8 ~ a1 + a3 + a5 + a7, data = trainData, family = binomial)
  testData$Predict = predict(t_mod,newdata=testData, type="response")
  testData$Check = as.factor(ifelse(testData$Predict > 0.5,1,0))
  #make a confusion table (which might only have 1 column)
  x = table(testData$a8,testData$Check)
  if (ncol(x) == 1) {
    if (colnames(x)[1] == "0") {
      x = cbind(x,"1"=c(0,0))
    } else {
      x = cbind("0"=c(0,0),x)
    }
  }
  x = x[2:1,2:1]
  accuracy = (x[1,1]+x[2,2])/nrow(testData)
  average_accuracy = average_accuracy + accuracy
  print(paste("Iteration i: ",i))
  print(x)
  print(paste("Accuracy: ",accuracy))
  print("--------------------")
}
#for 9 folds model: x is the confusion matrix,accuraccy is the accuracy and average_accuracy is the avarage accurracy.
print(paste("confusion matrix of 9folds:"))
print(x)
print(paste("accuracy of 9folds: ",accuracy))
print(paste("avaerage accuracy : ",average_accuracy))
print(paste("Average Accuracy with nfold=9 : ",average_accuracy/nfolds))


nfolds_1 = 14
#Randomly shuffle the data
dataset_shuffle_1<-ss[sample(nrow(ss)),]
#Create folds (makes a vector marking each fold for 14 folds)
folds_1 <- cut(seq(1,nrow(dataset_shuffle_1)),breaks=nfolds_1,labels=FALSE)
average_accuracy_n = 0.0
for(i in 1:nfolds_1){
  #Choose 1 fold and separate the testing and training data 
  testIndexes_1 <- which(folds_1==i,arr.ind=TRUE)
  testData_1 <- dataset_shuffle_1[testIndexes_1, ]
  trainData_1 <- dataset_shuffle_1[-testIndexes_1, ]
  #Build the model using glm of logistic regression with the highest accuracy model  
  t_mod_1 = glm(formula = a8 ~ a1 + a3 + a5 + a7, data = trainData_1, family = binomial)
  testData_1$Predict = predict(t_mod_1,newdata=testData_1, type="response")
  testData_1$Check = as.factor(ifelse(testData_1$Predict > 0.5,1,0))
  #make a confusion table (which might only have 1 column)
  y = table(testData_1$a8,testData_1$Check)
  if (ncol(y) == 1) {
    if (colnames(y)[1] == "0") {
      y = cbind(y,"1"=c(0,0))
    } else {
      y = cbind("0"=c(0,0),y)
    }
  }
  y = y[2:1,2:1]
  accuracy_n = (y[1,1]+y[2,2])/nrow(testData_1)
  average_accuracy_n = average_accuracy_n + accuracy_n
  print(paste("Iteration i: ",i))
  print(y)
  print(paste("Accuracy: ",accuracy_n))
  print("--------------------")
}
#for 14 folds model: y is the confusion matrix,accuraccy_n is the accuracy and average_accuracy_n is the avarage accurracy.
print(paste("confusion matrix of 14folds:"))
print(y)
print(paste("accuracy of 14folds: ",accuracy_n))
print(paste("avaerage accuracy : ",average_accuracy_n))
print(paste("Average Accuracy with nfold = 14: ",average_accuracy_n/nfolds_1))

#Save the new dataframe without the row names
write.csv(ss,"newdatafile.csv",row.names = FALSE)

#Questions:
#1: the training split 70-30 works best for me beacuse the accuraccy is relatively highers for the 5 modes when compared with the split done by 60-30 and also the highest accuracy is in split 70-30 for model 2 which is 0.7777778
#2: model two :t_2 in the split 70-30 worked best for me because the accuracy is 0.777778 which is the higest of all and the model has the columns a1,a3,a5,a7
#3: From the graph we could see the true positive is increasing and no curve in the false positive region,the area under the curve which is the accuracy is 0.81244 which is shows the true positive rate is high.
#4: the area under the curve is 0.8125 which is better as we could see the positive set belong to the match set is high and also the match cases are classified almost correctly. 
#5: 14folds is better when comapred with the 9 folds because the average accuracy with 14flods is 0.7210884 while the avarage accuracy with 9 folds is only 0.703109
