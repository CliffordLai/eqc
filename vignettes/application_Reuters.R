## ---- warning=FALSE,message=FALSE----------------------------------------
library(eqc)
library(tm)
library(NLP)
library(SnowballC)

## ------------------------------------------------------------------------
data(acq)
data(crude)

coall <- c(acq,crude)
coall <- tm_map(coall, removeNumbers) # remove digits
coall <- tm_map(coall,removePunctuation)
coall <- tm_map(coall,content_transformer(tolower))
coall <- tm_map(coall,stripWhitespace)
coall <- tm_map(coall,removeWords,stopwords("english"))
coall <- tm_map(coall,removeWords,stopwords("SMART"))
coall <- tm_map(coall,removeWords,stopwords("catalan"))
coall <- tm_map(coall,removeWords,stopwords("romanian"))
coall <- tm_map(coall,removeWords,stopwords("german"))
coall <- tm_map(coall,stemDocument)
tdm <- DocumentTermMatrix(coall)
dim(tdm)

# Construct data set
tdm <- as.matrix(tdm)
constVar <- (apply(tdm,2,sd)==0)

#Remove constant variables
tdm <- tdm[,!constVar]

cl.tdm <- c(rep(1,50),rep(2,20))
n <- nrow(tdm)
p <- ncol(tdm)

## ---- warning=FALSE------------------------------------------------------
# Fix the CV folds
set.seed(193)
nfolds <- 10
foldid <- numeric(n)
K <- 2
for(k in 1:K){
  indexk <- cl.tdm==k
  nk <- sum(indexk)
  foldid[indexk] <- sample(rep(1:nfolds,ceiling(nk/nfolds))[1:nk],size = nk,replace = FALSE)
}

# Tuning setting (Use a sparse set for illustration purpose)
nfolds_tuning <- 4
ncpu <- 1
seed_tuning <- 123
thetaList <- matrix(rep(seq(0.3,0.9,0.1),p),ncol=p)
lambda <- c(1,0.1,0.01,0.001,0.0001)
cost <-  c(0.5,1,2)

cvErr <- data.frame(fold=1:nfolds, MC=0, QC=0, EQCLSVM=0, EQCRidge=0, EQCLasso=0)

for(i in 1:nfolds){
  cat(i,"")
  # Extract CV folds
  train <- tdm[foldid!=i,]
  test <- tdm[foldid==i,]
  cl.train <- cl.tdm[foldid!=i]
  cl.test <- cl.tdm[foldid==i]

  #---------Model Fitting--------------#
  # Median-based classifier (MC)
  MC <- eqcTrain(train,cl.train,
                 thetaList=matrix(rep(0.5,p),nrow = 1),
                 method = "qc")
  predMC <- predict(MC,newdata = test,type = "class")[[1]]
  cvErr$MC[i] <-  mean(predMC!=cl.test)

  # Quantile-based classifier (QC)
  QC <- eqcTrain(train,cl.train,
                 thetaList=thetaList,
                 method = "qc",
                 tuneControl = list(nfolds=nfolds_tuning, fold.seed=seed_tuning,ncpu=ncpu))
  predQC <- predict(QC,newdata = test,type = "class")[[1]]
  cvErr$QC[i] <-  mean(predQC!=cl.test)


  # EQC/LSVM
  EQCLSVM <- eqcTrain(train,cl.train,
                       thetaList=thetaList,
                       method = "svm",
                       kernel = "linear",cost = cost,
                       tuneControl = list(nfolds=nfolds_tuning, fold.seed=seed_tuning,ncpu=ncpu))
  predEQCLSVM <- predict(EQCLSVM,newdata = test,type = "class")[[1]]
  cvErr$EQCLSVM[i] <-  mean(predEQCLSVM!=cl.test)

  # EQC/Ridge
  EQCRidge <- eqcTrain(train,cl.train,
                       thetaList=thetaList,
                       method = "glmnet",
                       alpha = 0,lambda = lambda,
                       tuneControl = list(nfolds=nfolds_tuning, fold.seed=seed_tuning,ncpu=ncpu),
                       lower.limits=0, upper.limits=Inf)
  predEQCRidge <- predict(EQCRidge,newdata = test,type = "class")[[1]]
  cvErr$EQCRidge[i] <-  mean(predEQCRidge!=cl.test)


  # EQC/Lasso
  EQCLasso <- eqcTrain(train,cl.train,
                       thetaList=thetaList,
                       method = "glmnet",
                       alpha = 1,lambda = lambda,
                       tuneControl = list(nfolds=nfolds_tuning, fold.seed=seed_tuning,ncpu=ncpu),
                       lower.limits=0, upper.limits=Inf)
  predEQCLasso <- predict(EQCLasso,newdata = test,type = "class")[[1]]
  cvErr$EQCLasso[i] <-  mean(predEQCLasso!=cl.test)
}

cvErr

colMeans(cvErr[,-1])

