# Functional-Linear-Regression
Research work on the application of functional linear regression as a variational problem and some applications to climate data


# Main Approach (Part 4, Morris):
Dimensionality reduction of the problem (a regression in finite dimensions) by projecting onto a family of orthogonal functions (Karhunen-Loève expansion), similar to principal component regression, Fourier, wavelets, signatures [2], or splines [1].

Regularization is applied in two ways: by truncating a basis of functions used in the Karhunen-Loève expansion, or by penalizing (OLS with added penalty, as in Lasso or ridge/Tikhonov).

Solution and approach use variational calculus techniques inspired by [3].

# Advantages of our method:

Simplicity: Similar to Tikhonov, compared to methods involving decomposition.
Lower Algorithmic Complexity: Tikhonov requires a matrix inversion, not this method.

# References

[1] C. Crambes, A. Kneip, and P. Sarda. Smoothing splines estimators for functional linear regression. Annals of statistics, 37(1):35–72, 2009.
[2] A. Fermanian. Functional linear regression with truncated signatures. Journal of multivariate
analysis, 192:105031, 2022.
[3] M. Garcin. Estimation of time-dependent Hurst exponents with variational smoothing and
application to forecasting foreign exchange rates. Physica A: statistical mechanics and its
applications, 483:462–479, 2017.
[4] J.S. Morris. Functional regression. Annual review of statistics and its application, 2:321–359,
2015.
