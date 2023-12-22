data {
  int n_players;

  int<lower=0> n_not_draws;
  array[n_not_draws, 2] int not_draws;

  int<lower=0> n_draws;
  array[n_draws, 2] int draws;
}

transformed data {
  real sigma = 1;
}

parameters {
  array[n_players] real mu;
  real<lower=0> t;
}

model {
  sigma ~ gamma(2, 0.5);
  t ~ exponential(0.5);
  mu ~ normal(0, 1);

  for (i in 1:n_not_draws) {
    real diff = mu[not_draws[i][1]+1] - mu[not_draws[i][2]+1];
    target += normal_lccdf(t | diff, sqrt(2)*sigma);
  }
  for (i in 1:n_draws) {
    real diff = mu[draws[i][1]+1] - mu[draws[i][2]+1];
    target += log_sum_exp(normal_lcdf(t | diff, sqrt(2)*sigma), -normal_lcdf(-t | diff, sqrt(2)*sigma));
  }
}
