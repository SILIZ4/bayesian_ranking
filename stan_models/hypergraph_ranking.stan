data {
  int n_games;
  int n_players;
  int N;
  array[n_games] int games_sizes;
  array[n_players] int players;
}

parameters {
  array[N] real mu;
  real<lower=0.1> sigma;
}

model {
  //sigma ~ gamma(2, 0.5);
  sigma ~ exponential(2);
  mu ~ normal(0, 1);

  int k=2;
  for (i in 1:n_games) {
    for (j in 2:games_sizes[i]) {
      target += normal_lccdf(mu[players[k]]
                              | mu[players[k-1]], sigma);
      k+=1;
    }
    k+=1;
  }
}
