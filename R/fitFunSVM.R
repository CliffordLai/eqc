#' @name  fitFunSVM
#' @aliases predFunSVM
#'
#' @title \code{fitFun} and \code{predFun} for a SVM
#'
#' @description
#' \code{fitFunSVM} defines the \code{fitFun} for fitting support vector mahines with
#' the \code{\link[e1071]{svm}}.
#' \code{predFunSVM} defines the corresponding \code{predFun}.
#'
#' @param trainQuant A \code{n*p} matrix containg \code{n} observations of \code{p} quantile-based transformed variables.
#' @param cl.train A vector of length \code{n} containing the class labels 1 or 2.
#' @param ... Further arguments to be passed to \code{\link[e1071]{svm}} for \code{fitFunSVM},
#' or \code{\link[e1071]{predict.svm}} for \code{predFunSVM}.
#'
#' @details \code{fitFunSVM} fits support vector mahines with \code{\link[glmnet]{glmnet}},
#' where the default settings except \code{family} are used.
#' To modify them, use the argument \code{...}.
#' See examples.
#'
#' @return \code{fitFunSVM} produces an object of class "svm".
#' See \code{\link[e1071]{svm}} for more details.
#'
#' @author Yuanhao Lai
#'
#' @seealso \code{\link{quantileTrain}}, \code{\link[e1071]{svm}}, \code{\link{fitFunGLMnet}}.
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
#' #-------------------------------------------------------------#
#' # Apply the linear SVM to the quantile-based transformed data #
#' #-------------------------------------------------------------#
#' thetaList <- rbind(rep(0.3,p),rep(0.4,p),rep(0.5,p))
#' ansListLSVM <- quantileTrain(train,cl.train,thetaList,
#'                               fitFun=fitFunSVM, kernel="linear",cost=c(0.5,1))
#' predListLSVM <- quantilePredict(ansListLSVM,newdata=test,
#'                                 predFun=predFunSVM)
#' acc <- sapply(predListLSVM,FUN = function(pred){colMeans(pred==cl.test)})
#' colnames(acc) <- c(0.3,0.4,0.5)
#' rownames(acc) <- c(0.5,1)
#' acc
#' #          0.3       0.4       0.5
#' #0.5 0.9829545 0.9659091 0.9715909
#' #1   0.9772727 0.9659091 0.9659091
#'
#' #-------------------------------------------------------------#
#' # Apply the radial SVM to the quantile-based transformed data #
#' #-------------------------------------------------------------#
#' thetaList <- rbind(rep(0.3,p),rep(0.4,p),rep(0.5,p))
#' ansListRSVM <- quantileTrain(train,cl.train,thetaList,
#'                               fitFun=fitFunSVM, kernel="radial",cost=1)
#' predListRSVM <- quantilePredict(ansListRSVM,newdata=test,
#'                                  predFun=predFunSVM)
#' acc <- sapply(predListRSVM,FUN = function(pred){colMeans(pred==cl.test)})
#' names(acc) <- c(0.3,0.4,0.5)
#' acc #0.9772727 0.9715909 0.9715909
#'
#'
#' #------------------------------------------------------------#
#' # Tune theta and cost in LSVM  using 5-fold cross validation #
#' #------------------------------------------------------------#
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
#' cost <- 2^(0:3)
#'
#' cvErr <- lapply(1:nfolds,FUN = function(fold){
#'   cvfold <- foldid==fold
#'   cvtrain <- train[!cvfold,]
#'   cvtest <- train[cvfold,]
#'   cl.cvtrain <- cl.train[!cvfold]
#'   cl.cvtest <- cl.train[cvfold]
#'   ansListLSVM <- quantileTrain(cvtrain,cl.cvtrain,thetaList,
#'                                fitFun=fitFunSVM, kernel="linear",cost=cost)
#'   predListLSVM <- quantilePredict(ansListLSVM,newdata=cvtest,
#'                                   predFun=predFunSVM)
#'   err <- sapply(predListLSVM,FUN = function(pred){colMeans(pred!=cl.cvtest)})
#'   return(err)
#' })
#' cvErr <- Reduce("+", cvErr)/nfolds
#' mincvErr <- min(cvErr,na.rm = TRUE)
#' minIndex <- which(cvErr==mincvErr,arr.ind = TRUE)
#' # Use the first one as the chosen parameters
#' thetaListC <- thetaList[minIndex[1,2],,drop=FALSE]
#' thetaListC[1] # theta=0.3
#' cost[minIndex[1,1]] # cost=2
#'
#' ## Fit the tuned LSVM
#' ansListLSVM <- quantileTrain(train,cl.train,thetaListC,
#'                              fitFun=fitFunSVM, kernel="linear",cost=cost[minIndex[1,1]])
#' predListLSVM <- quantilePredict(ansListLSVM,newdata=test,
#'                                 predFun=predFunSVM)
#' acc <-  sapply(predListLSVM,FUN = function(pred){mean(pred==cl.test)})
#' acc #0.9772727
#'
#' @export
#' @rdname fitFunSVM
#'
fitFunSVM <- function(trainQuant,cl.train,...){
  mySVM <- function(x,y,cost,...){lapply(cost, function(costi){e1071::svm(x,y,cost=costi,...)})}
  fit <- mySVM(trainQuant, factor(cl.train),...)
  return(fit)
}


#' @export
#' @rdname fitFunSVM
#'
#' @param fitFunReturn Fitted "svm" model object returned by \code{fitFunSVM}.
#' @param newdataQuant A matrix of \code{p} columns in which to look for
#' quantile-based transformed variables with which to predict.
#'
#' @return \code{predFunSVM} produces predicted classes given by \code{\link[e1071]{predict.svm}}.
#'
#' @keywords internal
#'
predFunSVM <- function(fitFunReturn,newdataQuant,...){
  pred <- sapply(fitFunReturn,function(fitFunReturnSub){predict(fitFunReturnSub,newdata=newdataQuant,...)})
  return(pred)
}


