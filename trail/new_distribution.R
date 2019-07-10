fMAL <- function(x,thetaV,qV,sV){
  exp(-sapply(x,function(x)sum(sV*(x-qV)*(thetaV-(x<qV)))))
}

# one
thetaV <- c(0.1)
qV <- c(2)
sV <- c(1)

curve(fMAL(x,thetaV,qV,sV),-10,10)

# two-parameter
thetaV <- c(0.1,0.7)
qV <- c(-1,2)
sV <- c(5,1)

curve(fMAL(x,thetaV,qV,sV),-10,10)

# three
thetaV <- c(0.1,0.3,0.8)
qV <- c(-1,2,4)
sV <- c(10,1,2)

curve(fMAL(x,thetaV,qV,sV),-10,10)

# three
thetaV <- c(0.1,0.3,0.8)
qV <- c(-1,2,4)
sV <- c(10,1,2)

curve(fMAL(x,thetaV,qV,sV),-10,10)


fx <- function(x){
  abs(x-q1)^3-abs(x-q2)^3
}

q1 <- 0.2
q2 <- q1+1


curve(fx,-5,5)
abline(h=0,lty=2,col="red")

uniroot(fx,c(-2,0))
uniroot(fx,c(-0.1,2))

#

library(ald)
n <- 10000
x <- rALD(n, mu = 4, sigma = 1, p = 0.1)
#x <- rnorm(n)
plot(density(x))

dx <- density(x)
#which.max(dx$y)
#dx$x[which.max(dx$y)]
#dx$y[which.max(dx$y)]
#mean(x)
#quantile(x,0.3)

library(modeest)
modeEst <- c(
  dx$x[which.max(dx$y)],
  mean(x),
  mlv(x, method = "lientz", bw = 0.2),
  mlv(x, method = "naive", bw = 1/3),
  mlv(x, method = "venter", type = "shorth"),
  mlv(x, method = "grenander", p = 4),
  mlv(x, method = "hrm", bw = 0.3),
  mlv(x, method = "hsm"),
  mlv(x, method = "parzen", kernel = "gaussian"),
  mlv(x, method = "tsybakov", kernel = "gaussian"),
  mlv(x, method = "asselin", bw = 2/3),
  mlv(x, method = "vieu"),
  mlv(x, method = "meanshift")
)

names(modeEst) <- c("kd","mean",
                    "lientz","naive","venter",
                    "grenander","hrm","hsm",
                    "parzen","tsybakov","asselin",
                    "vieu","meanshift")

modeEst
round(abs(modeEst-4)/4*100)/100


#
flogN <- function(x){
  (dnorm(log(x))/x) / (dnorm(log(x-0.2))/(x-0.2))
}

bayesC <- function(x){
  p1 <- (dnorm(log(x))/x)
  p2 <-  (dnorm(log(x-0.2))/(x-0.2))
  p2/(p1+p2)
}

library(ald)
bayesC_delinear <- function(x){
  # p1 <- (dnorm(log(x))/x)
  # p2 <-  (dnorm(log(x-0.2))/(x-0.2))

  #p1 <- dALD(x,p=0.2)
  #p2 <-  dALD(x-0.2,p=0.2)

  p1 <- dALD(x,p=0.2)
  p2 <-  dALD(x+1,p=0.1)

  # p1 <- dexp(x,rate = 2)
  # p2 <-  dexp(x+0.3,rate=2)

  nbp <- p2/(p1+p2)
  log(nbp/(1-nbp))
}


# curve(flogN,0.21,1)
#
# curve(bayesC,0.21,1)
# abline(h=0.5,col="red",lty=2)

x0 <- -10
x1 <- 11

curve(bayesC_delinear,x0,x1)
abline(h=0,col="red",lty=2)
abline(v=0)
abline(v=-0.3)

set.seed(193)
n <- 5000
sx <- sort(runif(n,x0,x1))
sy <- bayesC_delinear(sx)

lm1 <- lm(sy~sx+I(sx^2)+I(sx^3))

curve(bayesC_delinear,x0,x1)
abline(h=0,col="red",lty=2)
lines(sx,predict(lm1),lty=2,col="blue")

# abline(a=lm1$coefficients[1],
#        b=lm1$coefficients[2],
#        lty=2,col="blue")



#-------------------- Beta non-monotonic Bayes -------
a1 <- 0.5
b1 <- 0.6
a2 <- 2
b2 <- 3
curve(dbeta(x,a1,b1))
curve(dbeta(x,a2,b2),add=TRUE)


bayesBeta_delinear <- function(x){
  p1 <- dbeta(x,a1,b1)
  p2 <- dbeta(x,a2,b2)

  nbp <- p2/(p1+p2)
  log(nbp/(1-nbp))
}

x0 <- 0
x1 <- 1

curve(bayesBeta_delinear,x0,x1)
abline(h=0,col="red",lty=2)


rigidfun <- function(x,theta){
  q1 <- qbeta(theta,a1,b1)
  q2 <- qbeta(theta,a2,b2)

  (x-q1)*(theta-(x<q1)) - (x-q2)*(theta-(x<q2))
}

rigidfun2 <- function(x,theta1,theta2){
  rigidfun(x,theta1)+rigidfun(x,theta2)
}



rigidfun3 <- function(x,theta1,theta2,s=1){
  rigidfun(x,theta1)+s*rigidfun(x,theta2)
}

curve(rigidfun(x,theta = 0.7),x0,x1)
abline(h=0,col="red",lty=2)

curve(rigidfun2(x,0.2,0.5),x0,x1)
abline(h=0,col="red",lty=2)

curve(rigidfun3(x,0.2,0.5,2),x0,x1)
abline(h=0,col="red",lty=2)




#-------------------- Normal Laplace non-monotonic Bayes -------
library(ald)
x0 <- -7
x1 <- 7
curve(dnorm(x,1,sd=2),x0,x1)
curve(dALD(x,mu = 0,sigma = 0.9,p=0.4),x0,x1,add=TRUE)

# var of dALD, see mathematica


dnorm(6.8,3,sd=1)
dALD(6.8,mu = 0,sigma = 0.7,p=0.4)

bayes_delinear <- function(x){
  p1 <- dnorm(x,1,sd=2.2)
  p2 <- dALD(x,mu = 0,sigma = 0.7,p=0.4)

  nbp <- p2/(p1+p2)
  log(nbp/(1-nbp))
}

curve(bayes_delinear,x0,x1)
abline(h=0,col="red",lty=2)

#qALD(0.999,mu = 0,sigma = 0.4,p=0.4)

rigidfun <- function(x,theta){
  q1 <- qnorm(theta,1,sd=2.2)
  q2 <- qALD(theta,mu = 0,sigma = 0.7,p=0.4)

  (x-q1)*(theta-(x<q1)) - (x-q2)*(theta-(x<q2))
}

rigidfun2 <- function(x,theta1,theta2){
  rigidfun(x,theta1)+rigidfun(x,theta2)
}

rigidfun2a <- function(x,theta1,theta2,theta3){
  rigidfun(x,theta1)+rigidfun(x,theta2)+rigidfun(x,theta3)
}

rigidfun2b <- function(x,theta1,theta2,theta3,theta4){
  rigidfun(x,theta1)+rigidfun(x,theta2)+rigidfun(x,theta3)++rigidfun(x,theta4)
}

rigidfun3 <- function(x,theta1,theta2,s=1){
  rigidfun(x,theta1)+s*rigidfun(x,theta2)
}

curve(rigidfun(x,theta = 0.95),x0,x1)
abline(h=0,col="red",lty=2)

curve(rigidfun2(x,0.01,0.2),x0,x1)
abline(h=0,col="red",lty=2)

curve(rigidfun2b(x,0.01,0.1,0.9,0.99),x0,x1)
abline(h=0,col="red",lty=2)


curve(rigidfun3(x,0.2,0.5,2),x0,x1)
abline(h=0,col="red",lty=2)



#--------------------General non-monotonic Bayes -------
library(ald)

# Beta vs Beta
# x0 <- 0
# x1 <- 1
# density_class1 <- function(x){dbeta(x,0.5,0.6)}
# density_class2 <- function(x){dbeta(x,2,3)}
# quantile_class1 <- function(theta){qbeta(theta,0.5,0.6)}
# quantile_class2 <- function(theta){qbeta(theta,2,3)}
# rclass1 <- function(n){rbeta(n,0.5,0.6)}
# rclass2 <- function(n){rbeta(n,2,3)}


# laplace vs Normal
x0 <- -20
x1 <- 20
density_class1 <- function(x){dnorm(x,0.5,sd=1)}
density_class2 <- function(x){dALD(x,mu = 0,sigma = 1,p=0.3)}
quantile_class1 <- function(theta){qnorm(theta,0.5,sd=1)}
quantile_class2 <- function(theta){qALD(theta,mu = 0,sigma = 1,p=0.3)}
rclass1 <- function(n){rnorm(n,0.5,sd=1)}
rclass2 <- function(n){rALD(n,mu = 0,sigma = 1,p=0.3)}


# Beta vs Beta 2
# x0 <- 0
# x1 <- 1
# density_class1 <- function(x){dbeta(x,0.6,0.6)}
# density_class2 <- function(x){dbeta(x,2,3)}
# quantile_class1 <- function(theta){qbeta(theta,0.5,0.6)}
# quantile_class2 <- function(theta){qbeta(theta,2,3)}
# rclass1 <- function(n){rbeta(n,0.5,0.6)}
# rclass2 <- function(n){rbeta(n,2,3)}

curve(density_class1,x0,x1)
curve(density_class2,x0,x1,add=TRUE)

# var of dALD, see mathematica

bayes_delinear <- function(x){
  p1 <- density_class1(x)
  p2 <- density_class2(x)

  nbp <- p2/(p1+p2)
  log(nbp/(1-nbp))
}

BayeError <- integrate(f = function(x){
  0.5*(density_class1(x)<density_class2(x))*density_class1(x) +
    0.5*(density_class1(x)>density_class2(x))*density_class2(x)
},lower = x0,upper = x1)
curve(bayes_delinear,x0,x1,
      main=paste("Bayes decision boundary with error:",round(BayeError$value*1000)/1000,
                 "with absolute error",round(BayeError$abs.error*10000)/10000))
abline(h=0,col="red",lty=2)

#qALD(0.999,mu = 0,sigma = 0.4,p=0.4)

rigidfun <- function(x,theta){
  q1 <- quantile_class1(theta)
  q2 <- quantile_class2(theta)

  (x-q1)*(theta-(x<q1)) - (x-q2)*(theta-(x<q2))
}

rigidfun2 <- function(x,theta1,theta2){
  rigidfun(x,theta1)+rigidfun(x,theta2)
}

rigidfun2a <- function(x,theta1,theta2,theta3){
  rigidfun(x,theta1)+rigidfun(x,theta2)+rigidfun(x,theta3)
}

rigidfun3 <- function(x,theta1,theta2,s=1){
  rigidfun(x,theta1)+s*rigidfun(x,theta2)
}

curve(rigidfun(x,theta = 0.1),x0,x1)
abline(h=0,col="red",lty=2)

curve(rigidfun(x,theta = 0.9),x0,x1)
abline(h=0,col="red",lty=2)

curve(rigidfun2(x,0.1,0.9),x0,x1)
abline(h=0,col="red",lty=2)

# curve(rigidfun2a(x,0.01,0.1,0.9),x0,x1)
# abline(h=0,col="red",lty=2)
#
#
# curve(rigidfun3(x,0.2,0.5,2),x0,x1)
# abline(h=0,col="red",lty=2)


# Simulation experiment
set.seed(191)
n <- 500
ntest <- 200000
x <- c(rclass1(n),rclass2(n) )
y <- c(rep(1,n),rep(2,n))
#range(x)
xtest <- c(rclass1(ntest),rclass2(ntest) )
ytest <- c(rep(1,ntest),rep(2,ntest))
testErr <- numeric(4)


#1
glm1 <- glm(factor(y)~x,family = binomial())
curve(bayes_delinear,x0,x1)
abline(h=0,col="red",lty=2)
abline(a=glm1$coefficients[1], b=glm1$coefficients[2],lty=2,col="blue")
pred1 <- predict(glm1,data.frame(x=xtest), type = "response")
(testErr[1] <- mean(ifelse(pred1>0.5,2,1)!=ytest))


#2
x2 <- x^2
glm2 <- glm(factor(y)~x+x2,family = binomial())
curve(bayes_delinear,x0,x1)
abline(h=0,col="red",lty=2)
sX <- data.frame(x=sort(x),x2=sort(x)^2)
lines(sX$x, predict(glm2,sX),lty=2,col="blue")
pred2 <- predict(glm2,data.frame(x=xtest,x2=xtest^2), type = "response")
(testErr[2] <- mean(ifelse(pred2>0.5,2,1)!=ytest))


#3
thetaChoose <- c(0.1,0.9)
q1 <- quantile(x[1:n],thetaChoose)
q2 <- quantile(x[n+1:n],thetaChoose)
Qx1 <-(x-q1[1])*(thetaChoose[1]-(x<q1[1])) - (x-q2[1])*(thetaChoose[1]-(x<q2[1]))
Qx2 <-(x-q1[2])*(thetaChoose[2]-(x<q1[2])) - (x-q2[2])*(thetaChoose[2]-(x<q2[2]))
Qxtest1 <-(xtest-q1[1])*(thetaChoose[1]-(xtest<q1[1])) - (xtest-q2[1])*(thetaChoose[1]-(xtest<q2[1]))
Qxtest2 <-(xtest-q1[2])*(thetaChoose[2]-(xtest<q1[2])) - (xtest-q2[2])*(thetaChoose[2]-(xtest<q2[2]))

glm3 <- glm(factor(y)~Qx1+Qx2,family = binomial())
summary(glm3)
glm3$coefficients
curve(bayes_delinear,x0,x1)
abline(h=0,col="red",lty=2)
xorder <- order(x)
sX <- data.frame(Qx1=Qx1[xorder],Qx2=Qx2[xorder])
lines(x[xorder], predict(glm3,sX),lty=2,col="blue")
pred3 <- predict(glm3,data.frame(Qx1=Qxtest1,Qx2=Qxtest2), type = "response")
(testErr[3] <- mean(ifelse(pred3>0.5,2,1)!=ytest))

testErr
BayeError

# Try MARS
library(earth)
model_mars <- earth(factor(y)~x,degree = 2, glm=list(family=binomial))
pred1 <- predict(model_mars,data.frame(x=xtest), type = "response")
(testErr[4] <- mean(ifelse(pred1>0.5,2,1)!=ytest))

testErr




