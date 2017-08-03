import numpy as np
import matplotlib.pyplot as plt
import time
import pystan
import pymc3 as pymc
import edward
from pystan import StanModel

print("pystan version:", pystan.__version__)
print("pymc version:", pymc.__version__)
print("edward version:", edward.__version__)

np.random.seed(42)

all_num_points = [100, 10000]

all_alphas = [0, 1]
all_sigmas = [1, 3]
mu = 0

# Create the Stan model
fit_code = """
data {
	int<lower=0> N; 	// number of entities
	real y[N]; 		// normally distributed KPI in the control group
	real x[N]; 		// normally distributed KPI in the treatment group
}

parameters {
	real mu;			// population mean
	real<lower=0> sigma;// population variance
	real alpha;
}

transformed parameters {
	real delta;			// total effect size
	delta = alpha * sigma;
}

model {
	alpha ~ cauchy(0, 1);
	mu ~ cauchy(0, 1);
	sigma ~ gamma(2, 2);
	x ~ normal(mu+delta, sigma);
	y ~ normal(mu, sigma);
}
"""
stan_model = StanModel(model_code=fit_code)


def generate_data_normal(alpha, mu, sigma, N):
    delta = alpha * sigma
    xdata = np.random.normal(mu + delta, sigma, N)
    ydata = np.random.normal(mu, sigma, N)
    return (xdata, ydata)


def plot_data_hist(x, y):
    plt.figure(figsize=(10, 5))
    plt.hist(x, alpha=0.5, bins=500, label="X data")
    plt.hist(y, alpha=0.5, bins=500, label="Y data")
    plt.legend(loc='lower right', fontsize=11)
    plt.show()


def plot_trace_hist(trace):
    plt.hist(trace, alpha=0.5, bins=500, label="delta")
    plt.legend(loc='lower right', fontsize=11)
    plt.show()


def pystan_mcmc(xdata, ydata, num_points):
    fit_data = {'N': num_points, 'x': xdata, 'y': ydata}
    fit = stan_model.sampling(data=fit_data, iter=25000, chains=4, n_jobs=1, seed=1,
                      control={'stepsize': 0.01, 'adapt_delta': 0.99})
    # extract the traces
    traces = fit.extract()
    pystan_mcmc_trace = traces['delta']
    mean = pystan_mcmc_trace.mean()
    std = pystan_mcmc_trace.std()
    return mean, std


def pystan_vi(xdata, ydata, num_points):
    fit_data = {'N': num_points, 'x': xdata, 'y': ydata}
    results = stan_model.vb(data=fit_data, iter=10000)
    pystan_vi_trace = np.array(results['sampler_params'][3])
    mean = pystan_vi_trace.mean()
    std = pystan_vi_trace.std()
    return mean, std


def pymc3_mcmc(xdata, ydata):
    with pymc.Model() as model:
        alpha = pymc.Cauchy('alpha', 0, 1)
        mu = pymc.Cauchy('mu', 0, 1)
        sigma = pymc.Gamma('sigma', 2, 2)
        delta = pymc.Deterministic('delta', alpha * sigma)
        x = pymc.Normal('x', mu=mu+delta, sd=sigma, observed=xdata)
        y = pymc.Normal('y', mu=mu, sd=sigma, observed=ydata)

        # run the basic MCMC: we'll do 25000 iterations to match PyStan above
        trace = pymc.sample(25000, tune=500)
        pymc_trace = trace['delta']

        mean = pymc_trace.mean()
        std = pymc_trace.std()
    return mean, std


def pymc3_vi(xdata, ydata):
    with pymc.Model() as model:
        alpha = pymc.Cauchy('alpha', 0, 1)
        mu = pymc.Cauchy('mu', 0, 1)
        sigma = pymc.Gamma('sigma', 2, 2)
        delta = pymc.Deterministic('delta', alpha * sigma)
        x = pymc.Normal('x', mu=mu+delta, sd=sigma, observed=xdata)
        y = pymc.Normal('y', mu=mu, sd=sigma, observed=ydata)

        mean_field = pymc.fit(method='advi', n=10000)
        trace = mean_field.sample(25000)
        pymc_trace = trace['delta']

        mean = pymc_trace.mean()
        std = pymc_trace.std()
    return mean, std


def encodeDataKey(num_points, alpha, sigma, mu):
    key = str(num_points) + "," + str(alpha) + "," + str(sigma) + "," + str(mu)
    return key



start_time = time.time()

data_dict = {}
for num_points in all_num_points:
    for alpha in all_alphas:
        for sigma in all_sigmas:
            print("#data", num_points)
            print("alpha", alpha)
            print("delta", alpha * sigma)
            print("sigma", sigma)
            print("mu", mu)
            print("---------------------------")
            xdata, ydata = generate_data_normal(alpha, mu, sigma, num_points)
            data_key = encodeDataKey(num_points, alpha, sigma, mu)
            data_dict[data_key] = (xdata, ydata)






result = []
for num_points in all_num_points:
    for alpha in all_alphas:
        for sigma in all_sigmas:
            print("#data", num_points)
            print("alpha", alpha)
            print("delta", alpha * sigma)
            print("sigma", sigma)
            print("mu", mu)
            print("---------------------------")

            result_dict = {}
            result_dict['num_data'] = num_points
            result_dict['delta'] = alpha * sigma
            result_dict['sigma'] = sigma
            result_dict['mu'] = mu

            data_key = encodeDataKey(num_points, alpha, sigma, mu)
            (xdata, ydata) = data_dict[data_key]
            result_dict['true delta'] = xdata.mean() - ydata.mean()

            #TODO: why is true delta being modified????

            '''
            stan_mc_mean, stan_mc_std = pystan_mcmc(xdata, ydata, num_points)
            stan_vi_mean, stan_vi_std = pystan_vi(xdata, ydata, num_points)

            result_dict['delta_stan_mc_mean'] = stan_mc_mean
            result_dict['delta_stan_mc_std'] = stan_mc_std
            result_dict['delta_stan_vi_mean'] = stan_vi_mean
            result_dict['delta_stan_vi_std'] = stan_vi_std
            '''
            
            pymc3_mc_mean, pymc3_mc_std = pymc3_mcmc(xdata, ydata)
            pymc3_vi_mean, pymc3_vi_std = pymc3_vi(xdata, ydata)

            result_dict['delta_pymc3_mc_mean'] = pymc3_mc_mean
            result_dict['delta_pymc3_mc_std'] = pymc3_mc_std
            result_dict['delta_pymc3_vi_mean'] = pymc3_vi_mean
            result_dict['delta_pymc3_vi_std'] = pymc3_vi_std

            result.append(result_dict)

end_time = time.time()
print("--- total time %s seconds ---" % (end_time - start_time))


print(result)