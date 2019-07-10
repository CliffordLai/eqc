#' @name  quantileTrain
#' @aliases quantilePredict
#'
#' @title Train Classification Models with the Quantile-based Transformed Data
#'
#' @description
#' \code{quantileTrain} provides a neat way to train the classification model specified by \code{fitFun}
#' on a list of multiple quantile-based transformed data according to \code{thetaList}.
#' \code{quantilePredict} does the predictions based on the results of \code{quantileTrain} and the corresponding
#' prediction function \code{predFun}.
#'
#' @param train A \code{n*p} matrix containg \code{n} observations of \code{p} variables for training.
#' @param cl.train A vector of length \code{n} containing the class labels 1 or 2.
#' @param thetaList A matrix of \code{p} columns containing candidate probabilities of quantiles as rows.
#' @param fitFun A function takes quantile-based transformed data as input and trains a classifier. See details.
#' @param ... Further arguments to be passed to \code{fitFun} or \code{predictFun}.
#' @param skew.correct Skewness measures applied to correct the skewness direction of the variables.
#' The possibile choices are: Galton's skewness (default),
#' Kelley's skewness, the conventional skewness index based on the third standardized moment and
#' no correction.
#'
#' @details \code{quantileTrain} wraps up the general procedure of applying a classification model on
#' the quantile-based transformed data performed by \code{\link{quantileTransform}}.
#' The general procedure can be found from the example in \code{\link{quantileTransform}}.
#' An extra thing \code{quantileTrain} does is that
#' it will apply the classification method (\code{fitFun})
#' on every quantile-based transformed data with respect to each row of \code{thetaList}.
#'
#' The first two arguments of the function \code{fitFun} should be in the following format:
#'
#' \code{ function(trainQuant, cl.train, ...) }
#'
#' See the example for how to define \code{fitFun} for the logistic regression method.
#' There are also three built-in \code{fitFun}'s for fitting the penalized logistic regression model
#' \code{\link{fitFunGLMnet}}
#' and the SVM model \code{\link{fitFunSVM}} based on the packages \code{glmnet} and \code{e1071},
#' and \code{\link{fitFunQuantileDA}} for reproducing the quantile-based classifier.
#'
#' @return \code{quantileTrain} produces an object of class "quantTrain" is a list containing the following components:
#'
#' \item{fitFunReturnList}{A list where each component contains the return values of \code{fitFun}
#' applied on the quantile-transformed \code{X} with respect to each row of \code{thetaList}.}
#' \item{fitFun}{The input \code{fitFun}.}
#' \item{skew.correct}{The input type of skewness correction.}
#' \item{signSkew}{A vector of length \code{p} containing flip sign of each variable.}
#' \item{thetaList}{A matrix of p columns containing candidate probabilities of quantiles as rows.}
#' \item{qList}{A 3-D array where qList[k,,] (k=1,2,...) is a matrix of p columns containing
#' sample quantiles of Class k at each candidate probabilities.}
#' \item{zeroVar}{A list of positions of constant variables for each quantile-trasformed \code{X}
#' with respect to each row of \code{thetaList}.}
#'
#' @author Yuanhao Lai
#'
#' @references
#' Lai
#'
#' @seealso \code{\link{quantileTransform}}, \code{\link{fitFunGLMnet}},
#' \code{\link{fitFunSVM}},  \code{\link{fitFunQuantileDA}}.
#'
#' @keywords internal
#'
#' @examples
#' #----------------------------------------------------------------------#
#' # Apply the logistic regression to the quantile-based transformed data #
#' #----------------------------------------------------------------------#
#' # Function for fitting logistic regression to be passed to quantileTrain()
#' fitFunlogistic <- function(trainQuant,cl.train){
#'   fit <- glm(y~.,data=data.frame(y=factor(cl.train),trainQuant),family=binomial())
#'   return(fit)
#' }
#'
#' # Function for predicting class from the fitted logistic regression
#' # to be passed to quantilePredict()
#' predFunlogistic <- function(fitFunReturn,newdataQuant){
#'   pred <- predict(fitFunReturn,newdata=data.frame(newdataQuant),type="response")
#'   pred <- ifelse(pred<0.5,1,2)
#'   return(pred)
#' }
#'
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
#' # Fit the model with two sets of quantile-based transformations
#' thetaList <- rbind(rep(0.3,p),rep(0.4,p),rep(0.5,p))
#' ansList <- quantileTrain(train,cl.train,thetaList,fitFun=fitFunlogistic)
#'
#' # Evaluate the performance on the testing set by accuracy
#' predList <- quantilePredict(ansList,newdata=test,predFun=predFunlogistic)
#' acc <- sapply(predList,FUN = function(pred){mean(pred==cl.test)})
#' names(acc) <- c(0.3,0.4,0.5)
#' acc #0.9659091 0.9602273 0.9488636
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
#'   ansList <- quantileTrain(cvtrain,cl.cvtrain,thetaList,
#'                              fitFun=fitFunlogistic)
#'   predList <- quantilePredict(ansList,newdata=cvtest,
#'                                 predFun=predFunlogistic)
#'   err <- sapply(predList,FUN = function(pred){mean(pred!=cl.cvtest)})
#'   return(err)
#' })
#' cvErr <- Reduce("+", cvErr)/nfolds
#' mincvErr <- min(cvErr,na.rm = TRUE)
#' minIndex <- which(cvErr==mincvErr,arr.ind = TRUE)
#' # Use the first one as the chosen parameters
#' thetaListC <- thetaList[minIndex[1],,drop=FALSE]
#' thetaListC[1] # theta=0.3
#'
#' # Fit the tuned quantile-based classifier
#' ansList <- quantileTrain(train,cl.train,thetaListC,
#'                            fitFun=fitFunlogistic)
#' predList <- quantilePredict(ansList,newdata=test,
#'                               predFun=predFunlogistic)
#' acc <-  sapply(predList,FUN = function(pred){mean(pred==cl.test)})
#' acc #0.9659091
#'
#' @export
#' @rdname quantileTrain
quantileTrain <- function(train,
                          cl.train,
                          thetaList,
                          fitFun,...,
                          skew.correct=c("Galton","Kelley","skewness","none")){
  # Argument checking
  skew.correct <- match.arg(skew.correct)
  if(class(train)!="matrix") stop("Not correct object class of train!")
  K <- max(cl.train)
  if(any(sort(unique(cl.train))!=1:K))
    stop("cl.train should be a vector containing only(both) 1,2,...!")
  n <- nrow(train)
  p <- ncol(train)
  if(class(thetaList)!="matrix") stop("thetalist should be a matrix!")
  if(ncol(thetaList)!=p) stop("Incorrect dimension of thetaList!")
  fitFun <- match.fun(fitFun)
  if(any(names(formals(fitFun))[1:2]!=c("trainQuant","cl.train")))
    stop("Incorrect argument names of fitFun!")

  # Flip signs of variables to ensure same signs of skewness (Optional)
  train <- skewFlip(train,cl.train,skew.correct=skew.correct)
  signSkew <- train$signSkew
  train <- train$X

  # Compute sample quantiles based on the training set
  qList <- array(0,dim=c(K,dim(thetaList)))
  for(k in 1:K){
    classIndex <- cl.train==k
    for(j in 1:p){
      qList[k,,j] <- quantile(train[classIndex,j],thetaList[,j])
    }
  }


  # Begin modeling
  nT <- nrow(thetaList)
  zeroVar <- list()
  fitFunReturnList <- list()
  for(nt in 1:nT){
    # Perform the quantile-based transformation
    trainQuant <- quantileTransform(train,cl = NULL,thetaList[nt,],
                                    qList[1,nt,],qList[2,nt,])$qX

    # Remove variables with constant values to prevent failure of fitting algorithm
    varTrain <- apply(trainQuant,2,var)
    zeroVar[[nt]] <- which(varTrain==0)
    if(length(zeroVar[[nt]])!=0)
      warning("Transformed data contains constant variables!")
    trainQuant <- trainQuant[,setdiff(1:p,zeroVar[[nt]]),drop=FALSE]

    # Train the model
    fitFunReturnList[[nt]] <-  fitFun(trainQuant,cl.train,...)
  }

  ans <- list(fitFunReturnList=fitFunReturnList,
              fitFun=fitFun,
              skew.correct=skew.correct,
              signSkew=signSkew,
              thetaList=thetaList,
              qList=qList,
              zeroVar=zeroVar)
  class(ans) <- "quantTrain"
  return(ans)
}

#' @export
#' @rdname quantileTrain
#' @param object Object of class "quantTrain", created by \code{quantileTrain}.
#' @param newdata A matrix of \code{p} columns in which to look for variables with which to predict.
#' @param predFun A function does prediction on the quantile-transformed \code{newdata}
#' with the output of \code{fitFun}. See details
#'
#' @details  \code{quantilePredict} applies
#' prediction function \code{predFun} and the fitted model by \code{fitFun}
#' on every quantile-based transformed data with respect to each row of \code{thetaList}.
#'
#' The first two arguments of the function \code{predFun} should be in the following format:
#'
#' \code{ function(fitFunReturn,newdataQuant, ...) }
#'
#' @return \code{quantilePredict} produces a list where each component contains the return values of
#' \code{predFun} applied on the quantile-transformed \code{newdata} with respect to each
#' row of \code{thetaList} of \code{object}.
#'
#' @keywords internal
#'
quantilePredict <- function(object,newdata,predFun,...){
  if(class(object)!="quantTrain") stop("Wrong trained model class.")
  if(class(newdata)!="matrix") stop("Not correct object class of newdata!")
  predFun <- match.fun(predFun)
  if(any(names(formals(predFun))[1:2]!=c("fitFunReturn","newdataQuant")))
    stop("Incorrect argument names of predFun!")

  nT <- nrow(object$thetaList)
  p <- ncol(object$thetaList)
  newdata <- skewFlip(newdata,signSkew=object$signSkew)$X
  predFunReturnList <- list()
  for(nt in 1:nT){
    newdataQuant <- quantileTransform(newdata, cl=NULL,
                                      object$thetaList[nt,],
                                      object$qList[1,nt,],object$qList[2,nt,])$qX
    newdataQuant <- newdataQuant[,setdiff(1:p,object$zeroVar[[nt]])]
    predFunReturnList[[nt]] <- predFun(object$fitFunReturnList[[nt]],newdataQuant,...)
  }

  return(predFunReturnList)
}

