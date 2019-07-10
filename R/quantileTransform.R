#' @export
#'
#' @title Quantile-based transformation (Binary)
#'
#' @description
#' Compute the quantile-based transformed variables.
#' See details.
#'
#' @param X A \code{n*p} matrix containg \code{n} observations of \code{p} variables.
#' @param cl A vector of length \code{n} containing the class labels 1 or 2.
#' It must be provided if \code{q1} and \code{q2} are not provided
#' @param theta A vector of length \code{p} of probabilities that quantiles of \code{p} varibles are evaluated at.
#' @param q1 A vector of length \code{p} quantiles evaluated at \code{theta} for class 1. Default is NULL and
#' it will be obtained by computing sample quantiles with \code{\link{quantile}} on the observations where \code{cl=1}.
#' @param q2 A vector of length \code{p} quantiles evaluated at \code{theta} for class 2. Default is NULL and
#' it will be obtained by using \code{\link{quantile}} on the observations where \code{cl=2}.
#'
#' @details  For the \code{i}-th observation \eqn{x_{i}}, it computes a vector
#' \deqn{Q_{\theta}(x_{i})={
#' \Phi(x_{ij},\theta_j,q1_j)-\Phi(x_{ij},\theta_j,q2_j), j=1,...,p.} }
#' where \deqn{\Phi(x,\theta,q)=
#' [\theta+(1-2\theta)I\{x\le q\})] |x-q|.}
#' See examples for how the quantile-based transformation can be used to improve a classifier.
#'
#' @return A list of the following components:
#'
#' \item{qX}{A \code{n*p} matrix containing the quantile-based transformed \code{X}.}
#' \item{q1}{A vector of length \code{p} quantiles evaluated at \code{theta} for class 1.}
#' \item{q2}{A vector of length \code{p} quantiles evaluated at \code{theta} for class 2.}
#'
#' @author Yuanhao Lai
#'
#' @references
#' C. Hennig and  C. Viroli. Quantile-based classifiers.Biometrika, 103(2):435–446, 2016.
#'
#' @examples
#' #----------------------------------------------------------------------#
#' # Apply the logistic regression to the quantile-based transformed data #
#' #----------------------------------------------------------------------#
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
#' # Flip signs of variables to ensure same signs of skewness (Optional)
#' trainFlip <- skewFlip(train,cl.train,skew.correct="Galton")
#' signSkew <- trainFlip$signSkew
#' trainFlip <- trainFlip$X
#' testFlip <- skewFlip(test,signSkew=signSkew)$X
#'
#' # Compute sample quantiles based on the training set
#' theta <- rep(0.4,p)
#'
#' # Perform the quantile-based transformation
#' trainQuant <- quantileTransform(trainFlip, cl.train, theta) #the transformed training set
#' testQuant <- quantileTransform(testFlip, cl=NULL, theta,
#'                                q1=trainQuant$q1, q2=trainQuant$q2) #the transformed test set
#'
#' # Remove variables with constant values to prevent failure of fitting algorithm
#' varTrain <- apply(trainQuant$qX,2,var)
#' any(varTrain==0) #False, no need to remove any transformed variable
#'
#' # Perform the logistic regression on the training set and the transformed training set
#' ans0 <- glm(y~.,data=data.frame(y=factor(cl.train),train),family=binomial())
#' ans1 <- glm(y~.,data=data.frame(y=factor(cl.train),trainQuant$qX),family=binomial())
#'
#' # Evaluate the performance on the testing set by accuracy
#' pred0 <- ifelse(predict(ans0,newdata=data.frame(test),type="response")>0.5,2,1)
#' pred1 <- ifelse(predict(ans1,newdata=data.frame(testQuant$qX),type="response")>0.5,2,1)
#'
#' mean(pred0==cl.test) #raw: 0.9375
#' mean(pred1==cl.test) #transformed: 0.9602273
#'
quantileTransform <- function(X,cl=NULL,theta,q1=NULL,q2=NULL){
  if(class(X)!="matrix") stop("Not correct object class of X!")
  n <- nrow(X)
  p <- ncol(X)
  if(length(theta)!=p) stop("Incorrect length of theta!")

  # Process with quantiles for each class
  if(!is.null(cl)){
    if(any(sort(unique(cl))!=c(1,2)))
      stop("cl should be a vector containing only(both) 1 and 2!")
  }else{
    if(is.null(q1) | is.null(q2)) stop("q1 and q2 must be provided if cl is null!")
  }

  if(is.null(q1)){
    q1 <- sapply(1:p, FUN = function(j){
      quantile(X[cl==1,j],probs = theta[j])
    }) #For Class 1
  }else{
    if(length(q1)!=p) stop("Incorrect length of q1!")
  }

  if(is.null(q2)){
    q2 <- sapply(1:p, FUN = function(j){
      quantile(X[cl==2,j],probs = theta[j])
    }) #For Class 2
  }else{
    if(length(q2)!=p) stop("Incorrect length of q2!")
  }

  # Perform quantile-based transformation
  # qX <- matrix(0,n,p)
  # for(l in 1:p){
  #   thetal <- theta[l]
  #   q1l <- q1[l]
  #   q2l <- q2[l]
  #   qX[,l] <- (thetal+(1-2*thetal)*(X[,l]<q1l))*abs(X[,l]-q1l)-
  #     (thetal+(1-2*thetal)*(X[,l]<q2l))*abs(X[,l]-q2l)
  # }
  qX <- quantileDistC(X,theta,q1)-quantileDistC(X,theta,q2)
  colnames(qX) <- colnames(X)
  return(list(qX=qX,q1=q1,q2=q2))
}




