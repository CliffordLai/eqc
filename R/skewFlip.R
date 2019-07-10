#' @export
#'
#' @title Flip Signs of Variables based on Skewness Measures
#'
#' @description
#' Flip the signs of variables based on the specified skewness measures to ensure that
#' all varibales have non-negative skewnesses.
#'
#' @param X A \code{n*p} matrix containg \code{n} observations of \code{p} variables.
#' @param cl A vector of length \code{n} containing the class labels 1,2,....
#' @param signSkew sign of each variable to have a non-negative skewness. Default is NULL and
#' it will be determined by the skewness measure on \code{X}.
#' If \code{signSkew} is specified, then \code{cl} and \code{skew.correct} will be ignored.
#' @param skew.correct Skewness measures applied to correct the skewness direction of the variables.
#' The possibile choices are: Galton's skewness (default),
#' Kelley's skewness, the conventional skewness index based on the third standardized moment and
#' no correction..
#'
#' @return A list of the following two components:
#'
#' \item{X}{A \code{n*p} matrix with the sign of variables flipped to have positive skewnesses.}
#' \item{signSkew}{A vector of length \code{p} containing flip sign of each variable.}
#'
#' @author Yuanhao Lai
#'
#' @references
#' C. Hennig and  C. Viroli. Quantile-based classifiers.Biometrika, 103(2):435–446, 2016.
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
#'
#' # Flip signs of variables to ensure same sign of skewness in the traing set
#' trainFlip <- skewFlip(train,cl.train,skew.correct="Galton")
#' signSkew <- trainFlip$signSkew
#' trainFlip <- trainFlip$X
#'
#' # Apply the sign above to flip the test set
#' testFlip <- skewFlip(test,signSkew=signSkew)$X
#'
#'
skewFlip <- function(X,cl=NULL, signSkew=NULL,
                     skew.correct=c("Galton","Kelley","skewness","none")){
  skew.correct <- match.arg(skew.correct)
  n <- nrow(X)
  p <- ncol(X)

  if(is.null(signSkew)){
    if(is.null(cl)) stop("Required cl when signSkew is NULL!")
    if (skew.correct!="none") {
      K <- max(cl)
      if(any(sort(unique(cl))!= 1:K))
        stop("cl should be a vector containing only 1 to K!")
      skew <- matrix(0, K, p)
      for (i in 1:K) {
        skew[i, ] <- apply(X[cl == i, , drop = FALSE],
                           2, skewFun,skew.type=skew.correct)
      }
      totalSkew <- colSums(skew)/K
      totalSkew <- ifelse(is.na(totalSkew), 1, totalSkew)
      signSkew <- sign(totalSkew)
      signSkew[signSkew==0] <- 1
      X <- X * t(matrix(signSkew, p, n))
    }else{
      signSkew <- rep(1,p)
    }
  }else{
    if(length(signSkew)!=p){stop("Incorrect length of signSkew!")}
    X <- X * t(matrix(signSkew, p, n))
  }

  return(list(X=X,signSkew=signSkew))
}


#' @export
#'
#' @title Skewness Measure
#'
#' @description
#' Provide the computation of the Galton’s skewness measure, the Kelley skewness measure,
#' or the convention skewness measure.
#'
#' @param x A vector.
#' @param skew.type Skewness measures applied to correct the skewness direction of the variables.
#' The possibile choices are: Galton's skewness (default),
#' Kelley's skewness and the conventional skewness index based on the third standardized moment.
#'
#' @return A value of the specified skewness measure.
#'
#' @author Yuanhao Lai
#'
#' @references
#' C. Hennig and  C. Viroli. Quantile-based classifiers.Biometrika, 103(2):435–446, 2016.
#'
#' @keywords internal
#'
#' @examples
#' x <- rexp(100)
#' skewFun(x,skew.type="Galton")
#'
#'
skewFun <- function(x,skew.type=c("Galton","Kelley","skewness")){
  skew.type <- match.arg(skew.type)
  if(skew.type=="Galton"){
    quarts <- as.numeric(quantile(x, probs = c(0.25, 0.5, 0.75)))
    num <- quarts[1] + quarts[3] - 2 * quarts[2]
    denom <- quarts[3] - quarts[1]
    gskew <- num/denom
  }else if(skew.type=="Kelley"){
    quarts <- as.numeric(quantile(x, probs = c(0.1, 0.5, 0.9)))
    num <- quarts[1] + quarts[3] - 2 * quarts[2]
    denom <- quarts[3] - quarts[1]
    gskew <- num/denom
  }else if(skew.type=="skewness"){
    gskew <- mean((x - mean(x))^3)/(mean((x - mean(x))^2))^(3/2)
  }
  gskew
}

