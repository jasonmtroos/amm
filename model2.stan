functions {
  // calculate upper and lower bounds given x^+ and x or x and x^-, and
  // given dp = p^+ - p or dp = p - p^-
  real bound(real ux, real lx, real dp, real gamma, real alpha) {
    return log(dp * gamma) - log(alpha) - log(log1p(gamma * ux) - log1p(gamma * lx));
  }
}
data {
  // Number of decisions / observations
  int<lower = 0> T;
  
  // Observed quantities, prices, and available budgets
  real<lower = 0> x[T];
  real<lower = 0> p[T];
  real<lower = 0> M[T];
  
  // Indicators for cases when lb = -Inf or ub = Inf
  int <lower = 0, upper = 1> no_lb[T];
  int <lower = 0, upper = 1> no_ub[T];
  
  // Inferred next highest and next lowest quantities and prices
  real<lower = 0> ux[T];
  real<lower = 0> lx[T];
  real<lower = 0> up[T];
  real<lower = 0> lp[T];

  // Hyperparameters
  real<lower = 0> sigma_rate;
  real<lower = 0> alpha_rate;
  real<lower = 0> gamma_rate;
}
parameters {
  // Parameters on a natural unit scale
  real<lower = 0> sigma_raw;
  real<lower = 0> alpha_raw;
  real<lower = 0> gamma_raw;
}
transformed parameters {
  // Parameters on the model scale
  real sigma;
  real alpha;
  real gamma;
  
  sigma = sigma_raw / sigma_rate;
  alpha = alpha_raw / alpha_rate;
  gamma = 1 + gamma_raw / gamma_rate;
}
model {
  // Prior distributions for parameters on unit scale
  sigma_raw ~ gamma(2, .5);
  alpha_raw ~ exponential(1);
  gamma_raw ~ exponential(1);
  
  // Likelihood calculation, iterating over each observation
  for (t in 1:T) {
    real ub;
    real lb;
    
    // Determine lower bound (if any)
    if (no_lb[t]) {
      lb = negative_infinity();
    } else {
      lb = bound(x[t], lx[t], p[t] - lp[t], gamma, alpha);
    }
    
    // Determine upper bound (if any)
    if (no_ub[t]) {
      ub = positive_infinity();
    } else {
      ub = bound(ux[t], x[t], up[t] - p[t], gamma, alpha);
    }
    
    // Three cases for the likelihood calculation. 
    // 1. There is no lower bound but there is an upper bound
    //    The likelihood is F(ub) - F(-Inf) <=> F(ub) - 0 <=> F(ub)
    if (no_lb[t]) {
      target += normal_lcdf(ub | 0, sigma);
    } 
    // 2. There is no upper bound but there is a lower bound
    //    The likelihood if F(Inf) - F(lb) <=> 1 - F(lb)
    else if (no_ub[t]) { 
      target += normal_lccdf(lb | 0, sigma);
    } 
    // 3. There are both upper and lower bounds
    //    The likelihood is F(ub) - F(lb)
    else {
      real Fu;
      real Fl;
      Fu = normal_lcdf(ub | 0, sigma);
      Fl = normal_lcdf(lb | 0, sigma);
      target += log_diff_exp(Fu, Fl);
    }
  }
}
