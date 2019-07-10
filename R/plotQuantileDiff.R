#' @export
#'
#' @title Visualize the Difference between Quantiles of Two Classes
#'
#' @description
#' Visualize the abosulte standarized quantile differences between two classes.
#'
#' @param X A \code{n*p} matrix containg \code{n} observations of \code{p} variables for training.
#' @param Y A vector of length \code{n} containing the class labels 1 or 2.
#' @param theta A vector of candidate probabilities of quantiles.
#' @param method Whether the differences are shown in 2-dimension or in 3-dimension.
#' @param plot whether there is a plot or not.
#'
#'
#' @return A \code{p*T} matrix of the abosulte standarized quantile differences between two classes,
#' where \code{T} is the number of candidate quantiles.
#'
#' @author Yuanhao Lai
#'
#' @references
#' Lai
#'
#' @examples
#' # Read data
#' data(wdbc)
#' X <- as.matrix(wdbc[,-1])
#' Y <- wdbc[,1]
#'
#' # Visualize in a 2D heatmap
#' plotQuantileDiff(X,Y)
#'
#' # Visualize in 3D
#' plotQuantileDiff(X,Y,method="3D")
#'
plotQuantileDiff <- function(X,Y,
                             theta=seq(0.02,0.98,0.02),
                             method=c("2D","3D"),plot=TRUE){
  if(class(X)!="matrix") stop("Not correct object class of X!")
  p <- ncol(X)
  method <- match.arg(method)
  class1 <- Y==1
  class2 <- Y==2
  n1 <- sum(class1)
  n2 <- sum(class2)
  q1 <- apply(X[class1,],2,FUN = function(x){quantile(x,theta)})
  q2 <- apply(X[class2,],2,FUN = function(x){quantile(x,theta)})
  f1 <- apply(X[class1,],2,FUN = function(x){stats::approxfun(stats::density(x)) })
  f2 <- apply(X[class2,],2,FUN = function(x){stats::approxfun(stats::density(x)) })
  fq1 <- sapply(1:p,FUN = function(j){ f1[[j]](q1[,j]) })
  fq2 <- sapply(1:p,FUN = function(j){ f2[[j]](q2[,j]) })

  vq <- theta*(1-theta)*(1/(n1*fq1^2)+1/(n2*fq2^2))
  Tq <- (q2-q1)/sqrt(vq)
  zTq <- abs(t(Tq))

  if(plot){
    if(method=="2D"){
      fields::image.plot(1:p,theta,zTq, zlim=range(zTq),
                         col = rev(grDevices::heat.colors(12)),
                         xlab="variable",ylab="theta",
                         main="Abosulte Standarized Quantile Difference")
    }else if(method=="3D"){
      rgl::persp3d(1:p,theta,zTq,
                   col = "lightblue",
                   xlab="variable",ylab="theta",
                   zlab="Abosulte Standarized Quantile Difference")
    }
  }

  return(invisible(zTq))
}
