#' @name  fitFunQuantileDA
#' @aliases predFunQuantileDA
#'
#' @title \code{fitFun} and \code{predFun} for Reproducing Quantile-based Classifier
#'
#' @description
#' \code{fitFunQuantileDA} defines the \code{fitFun} for fitting
#' the quantile-based classifier (Hennig and Viroli, 2016).
#' \code{predFunQuantileDA} defines the corresponding \code{predFun}.
#'
#' @param trainQuant A \code{n*p} matrix containg \code{n} observations of \code{p} quantile-based transformed variables.
#' @param cl.train A vector of length \code{n} containing the class labels 1 or 2.
#' @param ... Further arguments to be passed to \code{\link[e1071]{svm}} for \code{fitFunSVM},
#' or \code{\link[e1071]{predict.svm}} for \code{predFunQuantileDA}.
#'
#' @details \code{fitFunQuantileDA} does nothing because estimation is not needed given
#' the quantile-based transformed data.
#' The decision is determined by the summation of all quantile-based transformed variables.
#'
#' @return \code{fitFunQuantileDA} produces a NULL object.
#'
#' @author Yuanhao Lai
#'
#' @references
#' C. Hennig and  C. Viroli. Quantile-based classifiers.Biometrika, 103(2):435–446, 2016.
#'
#' @seealso \code{\link{quantileTrain}}, \code{\link{fitFunSVM}}, \code{\link{fitFunGLMnet}}.
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
#' #-----------------------------------------#
#' # Reproduce the quantile-based classifier #
#' #-----------------------------------------#
#' thetaList <- rbind(rep(0.3,p),rep(0.4,p),rep(0.5,p))
#' ansListQC <- quantileTrain(train,cl.train,thetaList,
#'                               fitFun=fitFunQuantileDA)
#' predListLQC<- quantilePredict(ansListQC,newdata=test,
#'                                  predFun=predFunQuantileDA)
#' acc <- sapply(predListLQC,FUN = function(pred){colMeans(pred==cl.test)})
#' names(acc) <- c(0.3,0.4,0.5)
#' acc #0.8693182 0.9261364 0.9261364
#'
#'
#' #--------------------------------------------------------#
#' # Tune the parameter theta using 5-fold cross validation #
#' #--------------------------------------------------------#
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
#'
#' cvErr <- lapply(1:nfolds,FUN = function(fold){
#'   cvfold <- foldid==fold
#'   cvtrain <- train[!cvfold,]
#'   cvtest <- train[cvfold,]
#'   cl.cvtrain <- cl.train[!cvfold]
#'   cl.cvtest <- cl.train[cvfold]
#'   ansListQC <- quantileTrain(cvtrain,cl.cvtrain,thetaList,
#'                              fitFun=fitFunQuantileDA)
#'   predListQC <- quantilePredict(ansListQC,newdata=cvtest,
#'                                 predFun=predFunQuantileDA)
#'   err <- sapply(predListQC,FUN = function(pred){mean(pred!=cl.cvtest)})
#'   return(err)
#' })
#' cvErr <- Reduce("+", cvErr)/nfolds
#' mincvErr <- min(cvErr,na.rm = TRUE)
#' minIndex <- which(cvErr==mincvErr,arr.ind = TRUE)
#' # Use the first one as the chosen parameters
#' thetaListC <- thetaList[minIndex[1],,drop=FALSE]
#' thetaListC[1] # theta=0.4
#'
#' # Fit the tuned quantile-based classifier
#' ansListQC <- quantileTrain(train,cl.train,thetaListC,
#'                            fitFun=fitFunQuantileDA)
#' predListQC <- quantilePredict(ansListQC,newdata=test,
#'                               predFun=predFunQuantileDA)
#' acc <-  sapply(predListQC,FUN = function(pred){mean(pred==cl.test)})
#' acc #0.9261364
#'
#' @export
#' @rdname fitFunQuantileDA
#'
fitFunQuantileDA <- function(trainQuant,cl.train){
  return(NULL)
}


#' @export
#' @rdname fitFunQuantileDA
#'
#' @param fitFunReturn Object returned by \code{fitFunQuantileDA}.
#' @param newdataQuant A matrix of \code{p} columns in which to look for
#' quantile-based transformed variables with which to predict.
#'
#' @return \code{predFunQuantileDA} produces predicted classes of the quantile-based classifier.
#'
#' @keywords internal
#'
predFunQuantileDA <- function(fitFunReturn,newdataQuant){
  pred <- matrix((ifelse(rowSums(newdataQuant)>0,2,1)),ncol=1)
  return(pred)
}
