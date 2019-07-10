
meqcTrain  <- function(train,
                       cl.train,
                       thetaList,
                       lambda,
                       skew.correct=c("Galton","Kelley","skewness","none"),
                       postiveCoef=FALSE){
  # Argument checking
  skew.correct <- match.arg(skew.correct)
  if(class(train)!="matrix") stop("Not correct object class of train!")
  K <- max(cl.train)
  if(any(sort(unique(cl.train))!= 1:K))
    stop("cl.train should be a vector containing only 1 to K!")
  n <- nrow(train)
  p <- ncol(train)
  if(class(thetaList)!="matrix") stop("thetalist should be a matrix!")
  if(ncol(thetaList)!=p) stop("Incorrect dimension of thetaList!")

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
    # Perform the quantile-based transformation for multiclass
    trainQ <- matrix(0,nrow = n*K, ncol = p)
    Kstep <- seq(1,1+n*K-K,K)-1
    # for(k in 1:(K-1)){
    #   trainQ[k+Kstep,] <- -quantileTransform(train,cl = NULL,
    #                                          thetaList[nt,],
    #                                          qList[k,nt,],qList[K,nt,])$qX
    # }
    trainQK <- quantileDistC(train,thetaList[nt,],qList[K,nt,])
    for(k in 1:(K-1)){
      trainQ[k+Kstep,] <- -(quantileDistC(train,thetaList[nt,],qList[k,nt,])-trainQK)
    }
    Y <- matrix(0,K,n)
    for(i in 1:n){
      Y[cl.train[i],i] <- 1
    }

    # Remove variables with constant values to prevent failure of fitting algorithm
    varTrain <- apply(trainQ,2,var)
    zeroVar[[nt]] <- which(varTrain==0)
    if(length(zeroVar[[nt]])!=0)
      warning("Transformed data contains constant variables!")
    trainQ <- trainQ[,setdiff(1:p,zeroVar[[nt]]),drop=FALSE]

    # Train the model
    if(postiveCoef){
      #
      nlambda <- length(lambda)
      estimateBETA <- vector("list",nlambda)
      for(j in 1:nlambda){
        wrapNLL <- function(b1,...){
          w <- matrix(exp(b1),ncol = 1)
          nllw <- nlikmeqc(trainQ,Y,w,
                           lambda=lambda[j],...)
          #attr(nllw, "gradient") <- attr(nllw, "gradient")*as.vector(w)
          nllw[[2]] <- nllw[[2]]*as.vector(w)
          return(nllw)
        }

        # estimateBETA[[j]] <- stats::nlm(p= estimateBETA[[j-1]],f=wrapNLL,
        #                          print.level=0,iterlim=1000,
        #                          hessian = FALSE,check.analyticals=FALSE,
        #                          lambda=lambda[j])$estimate

        # estimateBETA[[j]] <- stats::nlm(p= rep(0,ncol(trainQ)),f=wrapNLL,
        #                          print.level=0,iterlim=1000,
        #                          hessian = FALSE,check.analyticals=FALSE,
        #                          lambda=lambda[j])$estimate
        estimateBETA[[j]] <- splitfngr::optim_share(par = rep(0,ncol(trainQ)),
                                                    fngr=wrapNLL, method = "BFGS")$par
      }

      fitFunReturnList[[nt]] <-  lapply(estimateBETA,exp)
    }else{
      #
      nlambda <- length(lambda)
      estimateBETA <- vector("list",nlambda)
      for(j in 1:nlambda){
        # estimateBETA[[j]] <- stats::nlm(p= estimateBETA[[j-1]],f=NLL,
        #                          print.level=0,iterlim=1000,
        #                          hessian = FALSE,check.analyticals=FALSE,
        #                          lambda=lambda[j])$estimate
        # estimateBETA[[j]] <- stats::nlm(p= rep(1,ncol(trainQ)),f=NLL,
        #                          print.level=0,iterlim=1000,
        #                          hessian = FALSE,check.analyticals=FALSE,
        #                          lambda=lambda[j])$estimate


        NLL <- function(w,...){
          nllw <- nlikmeqc(trainQ,Y,w,lambda=lambda[j],...)
          return(nllw)
        }
        estimateBETA[[j]] <- splitfngr::optim_share(par = rep(0,ncol(trainQ)),
                                                    fngr=NLL, method = "BFGS")$par
      }

      fitFunReturnList[[nt]] <-  estimateBETA
    }

  }

  ans <- list(fitFunReturnList=fitFunReturnList,
              skew.correct=skew.correct,
              signSkew=signSkew,
              thetaList=thetaList,
              qList=qList,
              zeroVar=zeroVar)
  return(ans)
}


nlikmeqc <- function(Q,Y,w,lambda=0){
  K <- dim(Y)[1]
  n <- dim(Y)[2]
  p <- dim(Q)[2]

  #Ensure a column vector
  w <- matrix(w,ncol = 1)
  #
  L <- Q%*%w   #dim(L) = c(n K,1)
  expQw <- exp(L)
  C <- t(matrix(1,1,K) %*% matrix(expQw,nrow = K)) #(n,1)

  #log-likelihood
  tvecY <- matrix(Y,nrow = 1)
  llw <- tvecY %*% L -sum(log(C))
  nllw <- -llw/n+lambda/2*sum(w^2)

  #gradient
  U <- Q*as.vector(expQw)  #(n K,p)
  A <- matrix(0,n,p) #(n,p)
  if(n<p){
    for(i in 1:n){
      index <- (i-1)*K
      A[i,] <- matrix(1,1,K) %*% U[index+1:K,]
    }
  }else{
    for(j in 1:p){
      A[,j] <- matrix(1,1,K) %*% matrix(U[,j],nrow=K)
    }
  }
  gllw <- tvecY %*% Q - colSums(A/as.vector(C)) #(1,p)
  gllw <- gllw/n
  gllw <- -gllw+as.vector(lambda*w)
  #attr(nllw, "gradient") <- gllw

  return(list(nllw,gllw))

  #Hessian requires a huge memory, sparse representation needed
  # ComputeHessian <- TRUE
  # if(ComputeHessian){
  #   FA <- A/as.vector(C)
  #   FA <- t(FA)%*%FA
  #   B2i <- matrix(rep(as.vector(1/C),each=K),nrow=1)
  #   for(j in 1:p){
  #     FA[j,] <- FA[j,]-B2i%*%(U*as.vector(Q[,j]))
  #   }
  #   diag(FA) <- diag(FA)-lambda
  #   attr(nllw, "hessian") <- -FA
  #
  #   # W1 <- as.vector(expQw) #W1 <- diag(as.vector(expQw))
  #   # W22 <- as.vector(1/C^2)  #diag(as.vector(1/C^2))
  #   # W3 <- rep(as.vector(1/C),each=K) #diag(rep(as.vector(1/C),each=K))
  #   # B1 <- kronecker(diag(n), matrix(1,1,K))
  #   # H <- t(B1) %*% W22 %*% B1-(1/W1)*W3
  #   # ggllw <- t(Q) %*% t(W1) %*% H %*% W1 %*% Q/n
  #
  #   # diag(ggllw) <- diag(ggllw)-lambda
  #   # attr(nllw, "hessian") <- -ggllw
  # }

  # return(nllw)
}


