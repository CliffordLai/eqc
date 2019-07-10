#' @name  fitFunGLMnet
#' @aliases predFunGLMnet
#'
#' @title \code{fitFun} and \code{predFun} for a Penalzied Logistic Regression
#'
#' @description
#' \code{fitFunGLMnet} defines the \code{fitFun} for fitting penalzied logistic regression models
#' with the \code{\link[glmnet]{glmnet}}.
#' \code{predFunGLMnet} defines the corresponding \code{predFun}.
#'
#' @param trainQuant A \code{n*p} matrix containg \code{n} observations of
#' \code{p} quantile-based transformed variables.
#' @param cl.train A vector of length \code{n} containing the class labels 1 or 2.
#' @param ... Further arguments to be passed to \code{\link[glmnet]{glmnet}} for \code{fitFunGLMnet},
#' or \code{\link[glmnet]{predict.glmnet}} for \code{predFunGLMnet}.
#'
#' @details \code{fitFunGLMnet} fits penalzied logistic regression models
#' with \code{\link[glmnet]{glmnet}},
#' where the default settings except \code{family} are used.
#' To modify them, use the argument \code{...}.
#' See examples.
#'
#' @return \code{fitFunGLMnet} produces an object of class "glmnet".
#' See \code{\link[glmnet]{glmnet}} for more details.
#'
#' @author Yuanhao Lai
#'
#' @seealso \code{\link{quantileTrain}}, \code{\link[glmnet]{glmnet}},
#' \code{\link{fitFunSVM}},\code{\link{fitFunQuantileDA}}.
#'
#' @keywords internal
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
#' #----------------------------------------------------------------------------#
#' # Apply the Lasso logistic regression to the quantile-based transformed data #
#' #----------------------------------------------------------------------------#
#' thetaList <- rbind(rep(0.3,p),rep(0.4,p),rep(0.5,p))
#' ansListLasso <- quantileTrain(train,cl.train,thetaList,
#'                               fitFun=fitFunGLMnet,
#'                               alpha=1,                           #Lasso penalty
#'                               lambda = c(2,1,0.1,0.01,0.001,0.0001),
#'                               lower.limits=0, upper.limits=Inf)
#' predListLasso <- quantilePredict(ansListLasso,newdata=test,
#'                                  predFun=predFunGLMnet,s=0.01) #Use lambda=0.01
#' acc <- sapply(predListLasso,FUN = function(pred){mean(pred==cl.test)})
#' names(acc) <- c(0.3,0.4,0.5)
#' acc # 0.9659091 0.9715909 0.9772727
#'
#' #----------------------------------------------------------------------------#
#' # Apply the Ridge logistic regression to the quantile-based transformed data #
#' #----------------------------------------------------------------------------#
#' thetaList <- rbind(rep(0.3,p),rep(0.4,p),rep(0.5,p))
#' ansListRidge <- quantileTrain(train,cl.train,thetaList,
#'                               fitFun=fitFunGLMnet,
#'                               alpha=0,                          #Ridge penalty
#'                               lambda = c(2,1,0.1,0.01,0.001,0.0001),
#'                               lower.limits=0, upper.limits=Inf)
#' predListRidge <- quantilePredict(ansListRidge,newdata=test,
#'                                  predFun=predFunGLMnet,s=0.01) #Use lambda=0.01
#' acc <- sapply(predListRidge,FUN = function(pred){mean(pred==cl.test)})
#' names(acc) <- c(0.3,0.4,0.5)
#' acc # 0.9659091 0.9772727 0.9772727
#'
#'
#' #------------------------------------------------------------------------------#
#' # Tune theta and lambda for the Ridge Regression using 5-fold cross validation #
#' #------------------------------------------------------------------------------#
#' # Set the fold
#' set.seed(117)
#' ntrain <- nrow(train)
#' p <- ncol(train)
#' foldid <- numeric(ntrain)
#' nfolds <- 5
#' index1 <- cl.train==1
#' index2 <- cl.train==2
#' n1 <- sum(index1)
#' n2 <- sum(index2)
#' foldid[index1] <- sample(rep(1:nfolds,ceiling(n1/nfolds))[1:n1],size = n1,replace = FALSE)
#' foldid[index2] <- sample(rep(1:nfolds,ceiling(n2/nfolds))[1:n2],size = n2,replace = FALSE)
#'
#' # Grid evaluate the CV-error
#' thetaList <- matrix(rep(c(0.1,0.2,0.3,0.4,0.5),p),ncol=p)
#' lambda <- c(3,1,0.1,0.01,0.001,0.0005,0.0001)
#'
#' cvErr <- lapply(1:nfolds,FUN = function(fold){
#'   cvfold <- foldid==fold
#'   cvtrain <- train[!cvfold,]
#'   cvtest <- train[cvfold,]
#'   cl.cvtrain <- cl.train[!cvfold]
#'   cl.cvtest <- cl.train[cvfold]
#'   ansListRidge <- quantileTrain(cvtrain,cl.cvtrain,thetaList,
#'                                 fitFun=fitFunGLMnet,
#'                                 alpha=0,
#'                                 lambda = lambda,
#'                                 lower.limits=0, upper.limits=Inf)
#'   predListRidge <- quantilePredict(ansListRidge,newdata=cvtest,
#'                                    predFun=predFunGLMnet)
#'   err <- sapply(predListRidge,FUN = function(pred){colMeans(pred!=cl.cvtest)})
#'   return(err)
#' })
#' cvErr <- Reduce("+", cvErr)/nfolds
#' mincvErr <- min(cvErr,na.rm = TRUE)
#' minIndex <- which(cvErr==mincvErr,arr.ind = TRUE)
#' # Use the first one as the chosen parameters
#' thetaListC <- thetaList[minIndex[1,2],,drop=FALSE]
#' thetaListC[1] # theta=0.5
#' lambda[minIndex[1,1]] # lambda=0.01
#'
#' # Fit the tuned RidgeQC
#' ansListRidge <- quantileTrain(train,cl.train,thetaListC,
#'                               fitFun=fitFunGLMnet,
#'                               alpha=0,
#'                               lambda = lambda,
#'                               lower.limits=0, upper.limits=Inf)
#' predListRidge <- quantilePredict(ansListRidge,newdata=test,
#'                                  predFun=predFunGLMnet,s=lambda[minIndex[1,1]])
#' acc <-  sapply(predListRidge,FUN = function(pred){mean(pred==cl.test)})
#' acc #0.9772727
#'
#' @export
#' @rdname fitFunGLMnet
#'
fitFunGLMnet <- function(trainQuant,cl.train,...){
  fit <- glmnet::glmnet(trainQuant, factor(cl.train),family = "binomial",...)
  return(fit)
}


#' @export
#' @rdname fitFunGLMnet
#'
#' @param fitFunReturn Fitted "glmnet" model object returned by \code{fitFunGLMnet}.
#' @param newdataQuant A matrix of \code{p} columns in which to look for
#' quantile-based transformed variables with which to predict.
#'
#' @return \code{predFunGLMnet} produces predicted classes converted from
#' the predicted probabilities given by \code{\link[glmnet]{predict.glmnet}}.
#'
#' @keywords internal
#'
predFunGLMnet <- function(fitFunReturn,newdataQuant,...){
  pred <- predict(fitFunReturn,newx=newdataQuant,type='response',...)
  pred <- ifelse(pred<0.5,1,2)
  return(pred)
}
