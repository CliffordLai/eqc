#' @export
#'
#' @title Train Ensemble Quantile Classifier
#'
#' @description
#' \code{eqcTrain} trains and tunes ensemble quantile classifiers.
#' Currently there are three ensemble ways.
#'
#' @param train A \code{n*p} matrix containg \code{n} observations of \code{p} variables for training.
#' @param cl.train A vector of length \code{n} containing the class labels 1,2,....
#' When there are more than two classes, \code{method} = "multiclass" should be used.
#' @param thetaList A matrix of \code{p} columns containing candidate probabilities of quantiles as rows.
#' @param method Ensemble method of \code{p} quantile-based classifiers. See details.
#' @param skew.correct Skewness measures applied to correct the skewness direction of the variables.
#' The possibile choices are: Galton's skewness (default),
#' Kelley's skewness, the conventional skewness index based on the third standardized moment and
#' no correction.
#' @param alpha The elasticnet mixing parameter used in \code{\link[glmnet]{glmnet}}, with \eqn{0\le \alpha \le 1}.
#' \code{alpha=1} is the lasso penalty, and \code{alpha=0} is the ridge penalty.
#' The default is 0.
#' @param lambda A user supplied lambda decreasing sequence used in \code{\link[glmnet]{glmnet}}
#' or for \code{method} = "multiclass".
#' The default is \code{c(0.5,0.1,0.005,0.001,0.0005,0.0001)}.
#' @param kernel The kernel used in training and predicting a SVM. The default is "linear".
#' @param cost A set of positive ‘C’-constants of the regularization term used in \code{\link[e1071]{svm}}.
#' The default is \code{c(0.5,1)}.
#' @param tuneControl A list of control parameters for tuning the model. See details.
#' @param ... Further arguments to be passed to \code{\link[glmnet]{glmnet}} or \code{\link[e1071]{svm}}.
#'
#' @details \code{eqcTrain} trains ensemble quantile classifiers
#' by applying selected linear classification method
#' on every quantile-based transformed data with respect to each row of \code{thetaList}.
#' An illurastration of how this work can be found in the example of \code{\link{quantileTransform}}.
#'
#' Currently, there are three ensemble methods specified by the argument \code{method}:
#' \code{"qc"} fits linear ensemble quantile classifier with equal weights.
#' \code{"glmnet"} fits linear ensemble quantile classifier with coefficients fitted by
#' a penalzied logistic regression \code{\link[glmnet]{glmnet}}.
#' \code{"svm"} fits linear ensemble quantile classifier with coefficients fitted by a SVM \code{\link[e1071]{svm}}.
#' \code{"multiclass"} fits linear ensemble quantile classifier with L2 regularized softmax rule for multiclass.
#'
#' The argument \code{tuneControl} specifies whether the model is tuned or not.
#' It is a list that can supply any of the following components:
#'
#' \code{nfolds}
#'
#' Number of folds - default is \code{NULL}, meaning no tuning.
#'
#' \code{fold.seed}
#'
#' An optional random seed for generating the folds.
#'
#' \code{foldid}
#'
#' An optional vector of values between 1 and \code{nfold} identifying what fold each observation is in.
#' If supplied, it overrides \code{nfold}.
#'
#' \code{type.measure}
#'
#' Loss used for cross-validation. The default is "me" for misclassification error,
#' or "auc" for the area under the ROC curve if \code{method} supports probablity outputs.
#'
#' \code{ncpu}
#'
#' number of compute nodes for doing the cross-validatoin with parallel. Default is 1.
#'
#' @return \code{eqcTrain} produces an object of class "eqcTrain" is a list containing the following components:
#'
#' \item{fitted}{A (large) list where each component contains the return values of the ensemble \code{method}
#' applied on the quantile-transformed \code{X} with respect to each row of \code{thetaList},
#' or the ensemble quantile classifer tuned by the cross-validation if \code{tuneControl} is specified.}
#' \item{K}{The number of classes.}
#' \item{method}{The ensemble method.}
#' \item{alpha}{The elasticnet mixing parameter used in \code{\link[glmnet]{glmnet}}.}
#' \item{lambda}{A lambda decreasing sequence used in \code{\link[glmnet]{glmnet}}.}
#' \item{kernel}{The kernel used in training and predicting a SVM.}
#' \item{cost}{A set of positive ‘C’-constants of the regularization term used in \code{\link[e1071]{svm}}.}
#' \item{skew.correct}{The input type of skewness correction.}
#' \item{thetaList}{A matrix of p columns containing candidate probabilities of quantiles as rows.}
#' \item{qList}{A 3-D array where qList[k,,] is a matrix of p columns containing
#' sample quantiles of Class k at each candidate probabilities.}
#' \item{signSkew}{A vector of length \code{p} containing flip sign of each variable.}
#' \item{zeroVar}{A list of positions of constant variables for each quantile-trasformed \code{X}
#' with respect to each row of \code{thetaList}.}
#' \item{CVmeasure}{A matrix consists of cross-validated \code{type.measure} for each combinations of tuning parameters, where
#' rows specify \code{theta} and columns specify \code{lambda} or \code{cost}.}
#' \item{cvparameter}{A list contains \code{theta}, \code{lambda} and \code{cost} selected by the cross-validation.}
#' \item{cvfold}{A list contains \code{nfolds}, \code{fold.seed}, \code{foldid} and \code{type.measure}.}
#'
#' @author Yuanhao Lai
#'
#' @references
#' Lai
#'
#' @seealso
#' \code{\link{quantileTransform}},
#' \code{\link[glmnet]{glmnet}}, \code{\link[e1071]{svm}}.
#'
#'
#' @examples
#' # Divide data into training set and test set randomly
#' data(wdbc)
#' set.seed(193)
#' trainIndex <- sample(c(rep(TRUE,2),rep(FALSE,1)),nrow(wdbc),replace=TRUE)
#' train <- as.matrix(wdbc[trainIndex,-1])
#' cl.train <- wdbc[trainIndex,1]
#' test <- as.matrix(wdbc[!trainIndex,-1])
#' cl.test <- wdbc[!trainIndex,1]
#' p <- ncol(train)
#'
#' # Tuning parameters
#' thetaList <- rbind(rep(0.3,p),rep(0.4,p),rep(0.5,p))
#'
#' # Fit the tuned EQC (QC)
#' fit1 <- eqcTrain(train,cl.train,
#'                  thetaList=thetaList,
#'                  method = "qc",
#'                  tuneControl = list(nfolds=5,fold.seed=117))
#' fit1$cvparameter #Selected tuning parameters
#' acc1 <- mean(predict(fit1,newdata = test,type = "class")[[1]]==cl.test)
#' acc1 #0.9261364
#'
#' # Fit the tuned EQC with a ridge logistic regression
#' lambda <- c(3,1,0.1,0.01,0.001,0.0005,0.0001)
#' fit2 <- eqcTrain(train,cl.train,
#'                  thetaList=thetaList,
#'                  method = "glmnet",
#'                  alpha = 0,lambda = lambda,
#'                  tuneControl = list(nfolds=5,fold.seed=117),
#'                  lower.limits=0, upper.limits=Inf)
#' fit2$cvparameter #Selected tuning parameters
#' acc2 <- mean(predict(fit2,newdata = test,type = "class")[[1]]==cl.test)
#' acc2 #0.9772727
#'
#' # Fit the tuned EQC with a linear SVM
#' cost <- 2^(0:3)
#' fit3 <- eqcTrain(train,cl.train,
#'                  thetaList=thetaList,
#'                  method = "svm",
#'                  kernel = "linear",cost = cost,
#'                  tuneControl = list(nfolds=5,fold.seed=117))
#' fit3$cvparameter #Selected tuning parameters
#' acc3 <- mean(predict(fit3,newdata = test,type = "class")[[1]]==cl.test)
#' acc3 #0.9772727
#'
#' #-----------------------Multiclass EQC-----------------------#
#' data(iris)
#' X <- as.matrix(iris[,1:4])
#' cl <- as.numeric(iris$Species)
#' n <- nrow(X)
#' p <- ncol(X)
#'
#' # Split train and test
#' set.seed(193)
#' trainIndex <- sample(c(TRUE,TRUE,FALSE),size = n,replace = TRUE)
#' train <- X[trainIndex,]
#' cl.train <- cl[trainIndex]
#' test <- X[!trainIndex,]
#' cl.test <- cl[!trainIndex]
#'
#' # Tuning parameters
#' thetaList <- rbind(rep(0.3,p),rep(0.4,p),rep(0.5,p))
#' lambda <- c(0.1,0.01)
#'
#' # Fit
#' fit4 <- eqcTrain(train,cl.train,
#'                  thetaList=thetaList,
#'                  method = "multiclass",
#'                  lambda = lambda,
#'                  tuneControl = list(nfolds=5,fold.seed=117))
#' fit4$cvparameter$thetaCV[1]
#' # Predict
#' pred4 <- predict(fit4,newdata = test,type = "class")[[1]]
#' acc4 <- mean(pred4==cl.test)
#' acc4 #0.9387755
#'
#' # Compared with SVM without tuning
#' model <- e1071::svm(x=train,y=factor(cl.train))
#' mean(predict(model,test)==cl.test) #0.9183673
#'
eqcTrain <- function(train,
                     cl.train,
                     thetaList,
                     method=c("qc","glmnet","svm","multiclass"),
                     skew.correct=c("Galton","Kelley","skewness","none"),
                     alpha=0,
                     lambda=c(0.5,0.1,0.005,0.001,0.0005,0.0001),
                     kernel=c("linear","polynomial","radial","sigmoid"),
                     cost=c(0.5,1),
                     ...,
                     tuneControl=list()){
  # Argument checking
  method <- match.arg(method)
  skew.correct <- match.arg(skew.correct)
  kernel <- match.arg(kernel)
  if(class(train)!="matrix") stop("Not correct object class of train!")
  K <- max(cl.train)
  if(any(sort(unique(cl.train))!=1:K))
    stop("cl.train should be a vector containing 1,2,...!")
  if(K>2 & (method!="qc" & method!="multiclass")){
    stop("There are more than 2 classes, method='qc' or method='multiclass' should be used!")
  }
  n <- nrow(train)
  p <- ncol(train)
  if(class(thetaList)!="matrix") stop("thetalist should be a matrix!")
  if(ncol(thetaList)!=p) stop("Incorrect dimension of thetaList!")
  if(alpha<0 | alpha>1) stop("Incorrect range of alpha!")

  # Without tuning
  if(length(tuneControl)==0){
    if(method=="qc"){
      fitted <- quantileTrain(train,cl.train,thetaList,
                              fitFun=fitFunQuantileDA,
                              skew.correct=skew.correct)
    }else if(method=="glmnet"){
      fitted <- quantileTrain(train,cl.train,thetaList,
                              fitFun=fitFunGLMnet,
                              alpha=alpha,
                              lambda = lambda,
                              skew.correct=skew.correct,...)
    }else if(method=="svm"){
      fitted <- quantileTrain(train,cl.train,thetaList,
                              fitFun=fitFunSVM,
                              kernel=kernel,cost=cost,
                              probability=TRUE,
                              skew.correct=skew.correct,...)
    }else if(method=="multiclass"){
      fitted <- meqcTrain(train,
                          cl.train,
                          thetaList,
                          lambda,
                          skew.correct=skew.correct,
                          postiveCoef=FALSE)
    }

    type.measure <- NA
    CVmeasure <- NA
    cvparameter <- NA
    cvfold <- NA
  }

  # Tune model with CV
  if(length(tuneControl)!=0){
    # Check type.measure and method
    type.measure <- tuneControl$type.measure
    if(is.null(type.measure)){
      type.measure <- "me"
    }else if(type.measure=="auc"){
      if(method=='qc' | method=='multiclass'){
        stop("method qc and multiclass do not support the evaluation of auc!")
      }
    }else if(type.measure!="me"){
      stop("Wrong input of tuneControl$type.measure!")
    }

    # Check CV options
    fold.seed <- NA
    if(is.null(tuneControl$nfolds)){
      if(is.null(tuneControl$foldid)){
        stop("tuneControl can only accepts nfolds or foldid!")
      }else{
        foldid <- tuneControl$foldid
        if(length(foldid)!=n){"incorrect length of foldid!"}
        nfolds <- max(foldid)
      }
    }else{
      if(is.null(tuneControl$foldid)){
        if(!is.null(tuneControl$fold.seed)){
          fold.seed <- tuneControl$fold.seed
          set.seed(fold.seed)
        }
        foldid <- numeric(n)
        nfolds <- tuneControl$nfolds
        for(k in 1:K){
          indexk <- cl.train==k
          nk <- sum(indexk)
          foldid[indexk] <- sample(rep(1:nfolds,ceiling(nk/nfolds))[1:nk],size = nk,replace = FALSE)
        }
      }else{
        #Simple check foldid
        foldid <- tuneControl$foldid
        nfolds <- max(foldid)
        if(length(foldid)!=n){"incorrect length of foldid!"}
        if(any(sort(unique(foldid))!=1:nfolds)){"mismatch of nfolds and foldid!"}
      }
    }

    # Start CV
    if(is.null(tuneControl$ncpu)){
      tuneControl$ncpu <- 1
    }else if(tuneControl$ncpu<1){
      stop("Incorrect number of clusters!")
    }
    tuneFun <- function(fold,...){
      cvfold <- foldid==fold
      cvtrain <- train[!cvfold,,drop=FALSE]
      cvtest <- train[cvfold,,drop=FALSE]
      cl.cvtrain <- cl.train[!cvfold]
      cl.cvtest <- cl.train[cvfold]


      cvfitted <- eqcTrain(cvtrain,
                           cl.cvtrain,
                           thetaList,
                           method=method,
                           skew.correct=skew.correct,
                           alpha=alpha,
                           lambda=lambda,
                           kernel=kernel,
                           cost=cost,...)
      if(type.measure=="me"){
        cvpredList <- predict(cvfitted,newdata=cvtest)
        cverr <- sapply(cvpredList,FUN = function(pred){colMeans(pred!=cl.cvtest)})
        return(cverr)
      }

      if(type.measure=="auc"){
        cvpredList <- predict(cvfitted,newdata=cvtest,type="probability")
        # aucApprox <- function(x,posIndex){
        #   mean(sample(x[posIndex],500,replace=T) > sample(x[!posIndex],500,replace=T))
        # }
        aucApprox <- function(scores,labels, N=1e5){
          pos <- sample(scores[labels], N, replace=TRUE)
          neg <- sample(scores[!labels], N, replace=TRUE)
          (sum(pos > neg) + sum(pos == neg)/2) / N # give partial credit for ties
        }
        posIndex <- cl.cvtest==2
        cvauc <- sapply(cvpredList,FUN = function(pred){apply(pred,2,aucApprox,labels=posIndex)})
        return(cvauc)
      }
    }


    if(tuneControl$ncpu==1){
      CVmeasure <- lapply(1:nfolds,FUN = tuneFun,...)
    }else{
      #use parLapply
      cl <- parallel::makeCluster(tuneControl$ncpu)
      parallel::clusterExport(cl, c("foldid", "train", "cl.train","eqcTrain","thetaList",
                                    "method","skew.correct","alpha","lambda","kernel","cost",
                                    "predict.eqcTrain"),
                              envir = environment())
      CVmeasure <- parallel::parLapply(cl,X=1:nfolds, fun=tuneFun,...)
      parallel::stopCluster(cl)
    }


    CVmeasure <- Reduce("+", CVmeasure)/nfolds
    optimCVmeasure <- ifelse(type.measure=="me",min(CVmeasure,na.rm = TRUE),max(CVmeasure,na.rm = TRUE))
    optimIndex <- which(CVmeasure==optimCVmeasure,arr.ind = TRUE)

    # Use the first one as the chosen parameters
    if(method=="qc"){
      thetaCV <- thetaList[optimIndex[1],,drop=FALSE]
      lambdaCV <- NA
      costCV <- NA
    }else if(method=="glmnet" | method=="multiclass"){
      if(length(lambda)==1){
        thetaCV <- thetaList[optimIndex[1],,drop=FALSE]
        lambdaCV <- lambda
        costCV <- NA
      }else if(nrow(thetaList)==1){
        thetaCV <- thetaList
        lambdaCV <- lambda[optimIndex[1]]
        costCV <- NA
      }else{
        thetaCV <- thetaList[optimIndex[1,2],,drop=FALSE]
        lambdaCV <- lambda[optimIndex[1,1]]
        costCV <- NA
      }
    }else if(method=="svm"){
      if(length(cost)==1){
        thetaCV <- thetaList[optimIndex[1],,drop=FALSE]
        lambdaCV <- NA
        costCV <- cost
      }else if(nrow(thetaList)==1){
        thetaCV <- thetaList
        lambdaCV <- NA
        costCV <- cost[optimIndex[1]]
      }else{
        thetaCV <- thetaList[optimIndex[1,2],,drop=FALSE]
        lambdaCV <- NA
        costCV <- cost[optimIndex[1,1]]
      }
    }

    cvparameter <- list(thetaCV=thetaCV, lambdaCV=lambdaCV, costCV=costCV)
    cvfold <- list(nfolds=nfolds, fold.seed=fold.seed, foldid=foldid, type.measure=type.measure)


    ans <- eqcTrain(train,
                    cl.train,
                    thetaCV,
                    method=method,
                    skew.correct=skew.correct,
                    alpha=alpha,
                    lambda=lambdaCV,
                    kernel=kernel,
                    cost=costCV,...)
    ans$type.measure <- type.measure
    ans$CVmeasure <- CVmeasure
    ans$cvparameter <- cvparameter
    ans$cvfold <- cvfold

    return(ans)
  }


  ans <- list(fitted=fitted$fitFunReturnList,
              K=K,
              method=method,
              alpha=alpha,
              lambda=lambda,
              kernel=kernel,
              cost=cost,
              skew.correct=skew.correct,
              thetaList=fitted$thetaList,
              qList=fitted$qList,
              signSkew=fitted$signSkew,
              zeroVar=fitted$zeroVar,
              type.measure= type.measure,
              CVmeasure=CVmeasure,
              cvparameter=cvparameter,
              cvfold=cvfold)
  class(ans) <- "eqcTrain"
  return(ans)
}

#' @export
#'
#' @title Predict Method for eqcTrain
#'
#' @param object Object of class "eqcTrain", created by \code{\link{eqcTrain}}.
#' @param newdata A matrix of \code{p} columns in which to look for variables with which to predict.
#' @param type Type of prediction required, there are two options "class" or "probability.
#' Default is the predicted class label.
#' @param ... Passing arguments to \code{\link[glmnet]{predict.glmnet}} or \code{\link[e1071]{predict.svm}}
#' depending on the ensemble method used by \code{object}.
#'
#' @return A list where each component contains the return values of
#' prediction methods applied on the quantile-transformed \code{newdata} with respect to each
#' row of \code{thetaList} of \code{object}.
#'
#' @examples
#' # Divide data into training set and test set randomly
#' data(wdbc)
#' set.seed(193)
#' trainIndex <- sample(c(rep(TRUE,2),rep(FALSE,1)),nrow(wdbc),replace=TRUE)
#' train <- as.matrix(wdbc[trainIndex,-1])
#' cl.train <- wdbc[trainIndex,1]
#' test <- as.matrix(wdbc[!trainIndex,-1])
#' cl.test <- wdbc[!trainIndex,1]
#' p <- ncol(train)
#'
#' # Tuning parameters
#' thetaList <- rbind(rep(0.3,p),rep(0.4,p),rep(0.5,p))
#'
#' # Fit the tuned EQC (QC)
#' fit1 <- eqcTrain(train,cl.train,
#'                  thetaList=thetaList,
#'                  method = "qc",
#'                  tuneControl = list(nfolds=5,fold.seed=117))
#'
#' # Prediction on the test set
#' pred1 <- predict(fit1,newdata = test,type = "class")[[1]]
#'
#'
predict.eqcTrain <- function(object, newdata,
                             type=c("class","probability"),...){
  if(class(newdata)!="matrix") stop("Not correct object class of newdata!")
  type <- match.arg(type)
  nT <- nrow(object$thetaList)
  p <- ncol(object$thetaList)
  newdata <- skewFlip(newdata,signSkew=object$signSkew)$X
  predFunReturnList <- list()

  # Determine the ensemble method
  if(object$method=="glmnet"){
    predType <- ifelse(type=="class","class","response")
    if(is.na(object$cvparameter[1])){
      predFun <- function(fitted,newdataQuant,...){
        pred <- predict(fitted,newx=newdataQuant,type=predType,...)
        return(pred)
      }
    }else{
      predFun <- function(fitted,newdataQuant,...){
        pred <- predict(fitted,newx=newdataQuant,type=predType,s=object$cvparameter$lambdaCV,...)
        return(pred)
      }
    }
  }else if(object$method=="svm"){
    if(type=="class"){
      predFun <- function(fitted,newdataQuant,...){
        pred <- sapply(fitted,function(fitFunReturnSub){
          predict(fitFunReturnSub,newdata=newdataQuant,probability=FALSE,...)})
        return(pred)
      }
    }else{
      predFun <- function(fitted,newdataQuant,...){
        pred <- sapply(fitted,function(fitFunReturnSub){
          tmp <- attr(predict(fitFunReturnSub,newdata=newdataQuant,probability=TRUE,...),'prob')
          tmp[,colnames(tmp)=="2"]})
        return(pred)
      }
    }
  }else if(object$method=="multiclass" | object$method=="qc"){
    # Perform the quantile-based transformation for multiclass
    K <- object$K
    for(nt in 1:nT){
      newdataQ <- matrix(0,nrow = nrow(newdata)*K, ncol = p)
      Kstep <- seq(1,1+nrow(newdata)*K-K,K)-1
      for(k in 1:(K-1)){
        newdataQ[k+Kstep,] <- -quantileTransform(newdata,cl = NULL,
                                                 object$thetaList[nt,],
                                                 object$qList[k,nt,],object$qList[K,nt,])$qX
      }
      newdataQ <- newdataQ[,setdiff(1:p,object$zeroVar[[nt]]),drop=FALSE]

      if(object$method=="multiclass"){
        wlambda <- object$fitted[[nt]]
      }else{
        wlambda <- list(rep(1,ncol(newdataQ)))
      }
      predProb <- lapply(wlambda, function(w){
        w <- matrix(w,ncol = 1)
        L <- newdataQ%*%w   #dim(L) = c(n K,1)
        expQw <- exp(L)
        C <- t(matrix(1,1,K) %*% matrix(expQw,nrow = K)) #(n,1)
        t(matrix(expQw,nrow = K)) / as.vector(C)  #(n,K) / (n,1)
      })

      if(type=="class")
        predProb <- matrix(sapply(predProb,function(predSub){apply(predSub,1,FUN = which.max)}),
                           ncol=length(wlambda))
      predFunReturnList[[nt]] <- predProb
    }
    return(predFunReturnList)
  }

  for(nt in 1:nT){
    newdataQuant <- quantileTransform(newdata, cl=NULL,
                                      object$thetaList[nt,],
                                      object$qList[1,nt,],object$qList[2,nt,])$qX
    newdataQuant <- newdataQuant[,setdiff(1:p,object$zeroVar[[nt]]),drop=FALSE]
    predFunReturnList[[nt]] <- predFun(object$fitted[[nt]],newdataQuant,...)
  }

  return(predFunReturnList)
}

