<!-- README.md is generated from README.Rmd. Please edit that file -->

eqc
===

It implements the ensemble quantile classifier (EQC) that extends the
quantile-based classifier (Hennig and Viroli, 2016b). An accompanying
vignette illustrates a simulation study and the application of the EQC
compared to the other methods.

Installation
------------

To install this package, run

``` r
if (!require("devtools")) install.packages("devtools")
devtools::install_github("CliffordLai/eqc")
```

Example
-------

This is a basic example which shows you how to train and tune the EQC.

``` r
library(eqc)
# Divide data into training set and test set randomly
data(wdbc)
set.seed(193)
trainIndex <- sample(c(rep(TRUE,2),rep(FALSE,1)),nrow(wdbc),replace=TRUE)
train <- as.matrix(wdbc[trainIndex,-1])
cl.train <- wdbc[trainIndex,1]
test <- as.matrix(wdbc[!trainIndex,-1])
cl.test <- wdbc[!trainIndex,1]
p <- ncol(train)

# Tuning parameters
thetaList <- rbind(rep(0.3,p),rep(0.4,p),rep(0.5,p))
# Fit the tuned EQC with a ridge logistic regression
lambda <- c(3,1,0.1,0.01,0.001,0.0005,0.0001)
fit <- eqcTrain(train,cl.train,
                 thetaList=thetaList,
                 method = "glmnet",
                 alpha = 0,lambda = lambda,
                 tuneControl = list(nfolds=5,fold.seed=117),
                 lower.limits=0, upper.limits=Inf)
fit$cvparameter #Selected tuning parameters
#> $thetaCV
#>      [,1] [,2] [,3] [,4] [,5] [,6] [,7] [,8] [,9] [,10] [,11] [,12] [,13]
#> [1,]  0.5  0.5  0.5  0.5  0.5  0.5  0.5  0.5  0.5   0.5   0.5   0.5   0.5
#>      [,14] [,15] [,16] [,17] [,18] [,19] [,20] [,21] [,22] [,23] [,24]
#> [1,]   0.5   0.5   0.5   0.5   0.5   0.5   0.5   0.5   0.5   0.5   0.5
#>      [,25] [,26] [,27] [,28] [,29] [,30]
#> [1,]   0.5   0.5   0.5   0.5   0.5   0.5
#> 
#> $lambdaCV
#> [1] 0.01
#> 
#> $costCV
#> [1] NA
acc <- mean(predict(fit,newdata = test,type = "class")[[1]]==cl.test)
acc  #Classification accuracy
#> [1] 0.9772727
```

Vignette
--------

The vignette include a simulation study and an application. List
vignettes in an HTML browser by,

``` r
browseVignettes(package = "eqc")
```
