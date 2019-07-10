#include <Rcpp.h>
#include <math.h>
using namespace Rcpp;

/* abs for singular */
// double abs3(double x) { return std::abs(x); }

//' @export
// [[Rcpp::export]]
NumericMatrix quantileDistC(NumericMatrix X,
                    NumericVector theta,
                    NumericVector q){
  int n = X.nrow();
  int p = X.ncol();
  NumericMatrix qX(n,p);

  for(int j=0; j<p; j++){
    for(int i=0; i<n; i++){
      if(X(i,j)<q[j]){
        // qX(i,j) = (theta(j) +(1-2*theta[j])*1)*abs3(X(i,j)-q(j));
        qX(i,j) = -(1-theta[j])*(X(i,j)-q(j));
      }else{
        // qX(i,j) = (theta(j))*abs3(X(i,j)-q(j));
        qX(i,j) = theta(j)*(X(i,j)-q(j));
      }
    }
  }

  return(qX);
}


