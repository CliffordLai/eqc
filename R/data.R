#' Processed Breast Cancer Wisconsin (Diagnostic) Data Set
#'
#' Data set of 569 observations with 31 variables obtained by removing the first columns (ID names)
#' of the original data set.
#'
#' @details The first column contains the Dignosis result (1 = benign, 2 = malignant).
#' The column 2 to column 31 (30 features) contain
#' ten types of real-valued features computed for each cell nucleus:
#'
#' a) radius (mean of distances from center to points on the perimeter)
#' b) texture (standard deviation of gray-scale values)
#' c) perimeter
#' d) area
#' e) smoothness (local variation in radius lengths)
#' f) compactness (perimeter^2 / area - 1.0)
#' g) concavity (severity of concave portions of the contour)
#' h) concave points (number of concave portions of the contour)
#' i) symmetry
#' j) fractal dimension ("coastline approximation" - 1)
#'
#' The mean, standard error, and "worst" or largest
#' (mean of the three largest values)
#' of these features were computed for each image,
#' resulting in 30 features.
#'
#' For instance, field 2 is Mean Radius, field
#' 12 is Radius SE, field 22 is Worst Radius.
#'
#' For more details, see the source page.
#'
#' @usage data(wdbc)
#'
#' @format A data.frame containing 569 observations of 31 variables.
#'
#' @source The raw data is downloaded from
#' \url{https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)}
#'
"wdbc"

