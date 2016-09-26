setwd("G:\\projects\\numerai stocks")
train<-read.csv(file.choose(),header = T,sep = ',')
test<-read.csv(file.choose(),header = T,sep = ',')

names(train)
names(test)
str(train)

y<-train$target
X_train<-train[,-c(16)]
X_test<-test[,-1]
##############################################
#xgboost
XGBoost <- function(X_train,y,X_test=data.frame(),cv=5,transform="none",objective="reg:linear",eta=0.1,max.depth=5,nrounds=50,gamma=0,min_child_weight=1,subsample=1,colsample_bytree=1,seed=123,metric="rmse",importance=0)
{
  # defining evaluation metric
  score <- function(a,b,metric)
  {
    switch(metric,
           accuracy = sum(abs(a-b)<=0.5)/length(a),
           auc = auc(a,b),
           logloss = -(sum(log(1-b[a==0])) + sum(log(b[a==1])))/length(a),
           mae = sum(abs(a-b))/length(a),
           precision = length(a[a==b])/length(a),
           rmse = sqrt(sum((a-b)^2)/length(a)),
           rmspe = sqrt(sum(((a-b)/a)^2)/length(a)))           
  }
  
  if (metric == "auc")
  {
    library(pROC)
  }
  
  cat("Preparing Data\n")
  X_train$order <- seq(1, nrow(X_train))
  X_train$result <- as.numeric(y)
  
  if (transform == "log")
  {
    X_train$result <- log(X_train$result)
  }
  
  # converting data to numeric
  for (i in 1:ncol(X_train))
  {
    X_train[,i] <- as.numeric(X_train[,i])
  }
  
  if (nrow(X_test)>0)
  {
    for (i in 1:ncol(X_test))
    {
      X_test[,i] <- as.numeric(X_test[,i])
    }    
  }
  
  X_train[is.na(X_train)] <- -1
  X_test[is.na(X_test)] <- -1
  
  X_test2 <- X_test
  
  set.seed(seed)
  X_train$randomCV <- floor(runif(nrow(X_train), 1, (cv+1)))
  
  # cross-validation
  cat(cv, "-fold Cross Validation\n", sep = "")
  for (i in 1:cv)
  {
    X_build <- subset(X_train, randomCV != i)
    X_val <- subset(X_train, randomCV == i)
    
    feature_names <- colnames(subset(X_build, select = -c(order, randomCV, result)))
    
    build <- as.matrix(subset(X_build, select = -c(order, randomCV, result)))
    val <- as.matrix(subset(X_val, select = -c(order, randomCV, result)))
    test <- as.matrix(X_test2)
    
    build_label <- as.matrix(subset(X_build, select = c('result')))
    
    # building model
    model_xgb <- xgboost(build,build_label,objective=objective,eta=eta,max.depth=max.depth,nrounds=nrounds,gamma=gamma,min_child_weight=min_child_weight,subsample=subsample,colsample_bytree=colsample_bytree,nthread=-1,verbose=0,eval.metric=metric)
    
    # variable importance
    if (importance == 1)
    {
      print (xgb.importance(feature_names=feature_names, model=model_xgb))
    }
    
    # predicting on validation data
    pred_xgb <- predict(model_xgb, val)
    if (transform == "log")
    {
      pred_xgb <- exp(pred_xgb)
    }
    
    X_val <- cbind(X_val, pred_xgb)
    
    # predicting on test data
    if (nrow(X_test) > 0)
    {
      pred_xgb <- predict(model_xgb, test)
      if (transform == "log")
      {
        pred_xgb <- exp(pred_xgb)
      }
    }
    
    cat("CV Fold-", i, " ", metric, ": ", score(X_val$result, X_val$pred_xgb, metric), "\n", sep = "")
    
    # initializing outputs
    if (i == 1)
    {
      output <- X_val
      if (nrow(X_test) > 0)
      {
        X_test <- cbind(X_test, pred_xgb)
      }      
    }
    
    # appending to outputs
    if (i > 1)
    {
      output <- rbind(output, X_val)
      if (nrow(X_test) > 0)
      {
        X_test$pred_xgb <- (X_test$pred_xgb * (i-1) + pred_xgb)/i
      }            
    }
    
    gc()
  } 
  
  # final evaluation score
  output <- output[order(output$order),]
  cat("\nXGBoost ", cv, "-Fold CV ", metric, ": ", score(output$result, output$pred_xgb, metric), "\n", sep = "")
  
  output <- subset(output, select = c("order", "pred_xgb"))
  
  # returning CV predictions and test data with predictions
  return(list("train"=output, "test"=X_test))  
}

model_xgb_1 <- XGBoost(X_train,y,X_test,cv=5,transform="none",objective="reg:linear",eta=0.1,max.depth=5,nrounds=50,gamma=0,min_child_weight=1,subsample=1,colsample_bytree=1,seed=123,metric="rmse",importance=0)
test_xgb_1 <- model_xgb_1[[2]]

submit <- data.frame(test$t_id, test_xgb_1$pred_xgb)
colnames(submit)<- c("t_id", "probability")
write.csv(submit,"Predictions3_xgb.csv",row.names=F) 
#############################################################
#gbm
GBMRegression<- function(X_train,y,X_test=data.frame(),cv=5,distribution="gaussian",n.trees=50,n.minobsinnode=5,interaction.depth=2,shrinkage=0.001,seed=123,metric="rmse")
{
  # defining evaluation metric
  score <- function(a,b,metric)
  {
    switch(metric,
           mae = sum(abs(a-b))/length(a),
           rmse = sqrt(sum((a-b)^2)/length(a)),
           rmspe = sqrt(sum(((a-b)/a)^2)/length(a)))
  }
  
  cat("Preparing Data\n")
  X_train$order <- seq(1, nrow(X_train))
  X_train$result <- as.numeric(y)
  
  set.seed(seed)
  X_train$randomCV <- floor(runif(nrow(X_train), 1, (cv+1)))
  
  # cross-validation
  cat(cv, "-fold Cross Validation\n", sep = "")
  for (i in 1:cv)
  {
    X_build <- subset(X_train, randomCV != i, select = -c(order, randomCV))
    X_val <- subset(X_train, randomCV == i) 
    
    # building model
    model_gbm <- gbm(result ~.,data=X_build,distribution=distribution,n.trees=n.trees,n.minobsinnode=n.minobsinnode,interaction.depth=interaction.depth,shrinkage=shrinkage)
    
    # predicting on validation data
    pred_gbm <- predict(model_gbm, X_val, n.trees)
    X_val <- cbind(X_val, pred_gbm)
    
    # predicting on test data
    if (nrow(X_test) > 0)
    {
      pred_gbm <- predict(model_gbm, X_test, n.trees)
    }
    
    cat("CV Fold-", i, " ", metric, ": ", score(X_val$result, X_val$pred_gbm, metric), "\n", sep = "")
    
    # initializing outputs
    if (i == 1)
    {
      output <- X_val
      if (nrow(X_test) > 0)
      {
        X_test <- cbind(X_test, pred_gbm)
      }      
    }
    
    # appending to outputs
    if (i > 1)
    {
      output <- rbind(output, X_val)
      if (nrow(X_test) > 0)
      {
        X_test$pred_gbm <- (X_test$pred_gbm * (i-1) + pred_gbm)/i
      }            
    }
    
    gc()
  } 
  
  # final evaluation score
  output <- output[order(output$order),]
  cat("\nGBM ", cv, "-Fold CV ", metric, ": ", score(output$result, output$pred_gbm, metric), "\n", sep = "")
  
  output <- subset(output, select = c("order", "pred_gbm"))
  
  # returning CV predictions and test data with predictions
  return(list(output, X_test))  
}
model_gbm <- GBMRegression(X_train,y,X_test,cv=5,distribution="gaussian",n.trees=50,n.minobsinnode=5,interaction.depth=2,shrinkage=0.001,seed=123,metric="rmse")
test_gbm <- model_gbm[[2]]
submit <- data.frame(test$t_id, test_gbm$pred_gbm)
colnames(submit)<- c("t_id", "probability")
write.csv(submit,"Predictions2_gbm.csv",row.names=F) 
#################################################################
#correlation
# load the library
library(mlbench)
library(caret)

# calculate correlation matrix
correlationMatrix <- cor(X_train[,-15])
# summarize the correlation matrix
print(correlationMatrix)
# find attributes that are highly corrected (ideally >0.75)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.5)
# print indexes of highly correlated attributes
print(highlyCorrelated)

model <- train(target~., data=train, method="lvq", preProcess="scale", trControl=control)

####################################################################
x<-apply(X_train[,-15], MARGIN = 2, FUN = function(X) (X - min(X))/diff(range(X)))
w<-apply(X_test[,-15], MARGIN = 2, FUN = function(X) (X - min(X))/diff(range(X)))

X_train<-data.frame(x)
X_test<-data.frame(w)
#######################################################################3

model<-C5.0( target~., train[,-c(16)],rules = T)
summary(model)
str(train)
train$target<-as.factor(train$target)
##################################################3

X_train<-subset(train,select=c(f5,c1,f6,f8,target))
X_test<-subset(test,select=c(f5,c1,f6,f8))
model_xgb_1 <- XGBoost(X_train,y,X_test,cv=5,transform="none",objective="reg:linear",eta=0.1,max.depth=5,nrounds=50,gamma=0,min_child_weight=1,subsample=1,colsample_bytree=1,seed=123,metric="rmse",importance=0)
test_xgb_1 <- model_xgb_1[[2]]

submit <- data.frame(test$t_id, test_xgb_1$pred_xgb)
colnames(submit)<- c("t_id", "probability")
write.csv(submit,"Predictions4_xgb.csv",row.names=F) 
#####################################################
model_glm<-glm(target~., family = binomial, data = X_train)
summary(model_glm)
Prediction <- predict(model_glm, X_test)
h<-apply(Prediction, MARGIN = 2, FUN = function(X) (X - min(X))/diff(range(X)))
g<-data.frame(Prediction)
for (i in 1:nrow(g))
{
  if (g[i,]<0) {g[i,] = g[i,]*-1}
 
}

g<-data.frame(g)
submit <- data.frame(test$t_id, g$Prediction)
colnames(submit)<- c("t_id", "probability")
write.csv(submit,"Predictions5_glm.csv",row.names=F) 

