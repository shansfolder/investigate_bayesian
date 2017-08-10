import numpy as np
import matplotlib.pyplot as plt
import time

import pandas as pd
import pystan
import pymc3 as pymc
import edward as ed
import tensorflow as tf
from edward.models import Normal, Gamma, StudentT, Empirical
from edward.models import NormalWithSoftplusScale, GammaWithSoftplusConcentrationRate
import os
from pystan import StanModel

print("pystan version:", pystan.__version__)
print("pymc version:", pymc.__version__)
print("edward version:", ed.__version__)

np.random.seed(42)

all_num_points = [100, 10000]

all_alphas = [0, 1]
all_sigmas = [1, 3]
mu = 0

# Create the Stan model
fit_code = """
data {
	int<lower=0> Ny; 	// number of entities in the control group
	int<lower=0> Nx; 	// number of entities in the treatment group
	real y[Ny]; 		// normally distributed KPI in the control group
	real x[Nx]; 		// normally distributed KPI in the treatment group
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

def read_csv(f_csv):
    if os.path.isfile(f_csv):
        with open(f_csv, 'r') as f_data:
            bq_df = pd.read_csv(f_data)
            return bq_df
    else:
        print('Data does not exist!')

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


def pystan_mcmc(xdata, ydata, nx, ny):
    fit_data = {'Ny': ny, 'Nx': nx, 'x': xdata, 'y': ydata}
    fit = stan_model.sampling(data=fit_data, iter=25000, chains=4, n_jobs=1, seed=1,
                      control={'stepsize': 0.01, 'adapt_delta': 0.99})
    # extract the traces
    traces = fit.extract()
    pystan_mcmc_trace = traces['delta']
    mean = pystan_mcmc_trace.mean()
    std = pystan_mcmc_trace.std()
    return mean, std


def pystan_vi(xdata, ydata, nx, ny):
    fit_data = {'Ny': ny, 'Nx': nx, 'x': xdata, 'y': ydata}
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


def edward_vi(xdata, ydata, nx, ny):
    sess = ed.get_session()

    # FORWARD MODEL, Prior
    mu = StudentT(1.0, [0.0], [1.0])
    delta = StudentT(1.0, [0.0], [1.0])
    sigma = Gamma([2.0], [2.0])

    x = Normal(tf.tile(mu + delta, [nx]), tf.tile(sigma, [nx]))
    y = Normal(tf.tile(mu, [ny]), tf.tile(sigma, [ny]))

    '''
    Mean and delta are best approximated by the NormalWithSoftplusScale distribution
    with the softplus function on the scale (sigma)(variance should be positive) parameter 
    since Cauchy and Normal are both defined on positive and negative scales.
    Sigma as a variance should be always positive: 
    we approximate sigma with GammaWithSoftplusConcentrationRate distribution 
    ensuring the positive concentration and rate parameters.
    '''
    # BACKWARD MODEL
    q_mu = NormalWithSoftplusScale(loc=tf.Variable([0.0]), scale=tf.Variable([1.0]))
    q_sigma = GammaWithSoftplusConcentrationRate(tf.nn.softplus(tf.Variable([1.0])), tf.nn.softplus(tf.Variable([1.0])))
    q_delta = NormalWithSoftplusScale(loc=tf.Variable([0.0]), scale=tf.Variable([1.0]))

    # INFERENCE
    inference = ed.KLqp({delta: q_delta, mu: q_mu, sigma: q_sigma}, data={x: xdata, y: ydata})
    inference.run(n_iter=20000, n_print=200, n_samples=10)

    T = 10000
    q_delta_sample = sess.run(q_delta.sample(sample_shape=T))

    mean = q_delta_sample.mean()
    std = q_delta_sample.std()
    return mean, std


def encodeDataKey(num_points, alpha, sigma, mu):
    key = str(num_points) + "," + str(alpha) + "," + str(sigma) + "," + str(mu)
    return key


def addDerivedKPIColumn(dataframe, derived_kpi_name, numerator_column, denominator_column):
    ctrl_reference_kpis = dataframe.loc[dataframe.variant == 'Control', denominator_column]
    treat_reference_kpis = dataframe.loc[dataframe.variant == 'Treatment', denominator_column]

    n_nan_ref_ctrl = sum(ctrl_reference_kpis == 0) + np.isnan(ctrl_reference_kpis).sum()
    n_non_nan_ref_ctrl = len(ctrl_reference_kpis) - n_nan_ref_ctrl

    n_nan_ref_treat = sum(treat_reference_kpis == 0) + np.isnan(treat_reference_kpis).sum()
    n_non_nan_ref_treat = len(treat_reference_kpis) - n_nan_ref_treat

    ctrl_weights = n_non_nan_ref_ctrl * ctrl_reference_kpis / np.nansum(ctrl_reference_kpis)
    treat_weights = n_non_nan_ref_treat * treat_reference_kpis / np.nansum(treat_reference_kpis)

    newColumn = {derived_kpi_name: dataframe[numerator_column] / dataframe[denominator_column]}
    dataframe = dataframe.assign(**newColumn)
    dataframe.loc[dataframe.variant == 'Control', derived_kpi_name] *= ctrl_weights
    dataframe.loc[dataframe.variant == 'Treatment', derived_kpi_name] *= treat_weights

    n_nan = np.isnan(dataframe[derived_kpi_name]).sum()
    nan_percentage_str = "%.4f" % (n_nan / len(dataframe))
    msg = derived_kpi_name + ": " + str(n_nan) + " out of " + str(len(dataframe)) + \
          " is nan. Percentage:" + nan_percentage_str
    print(msg)
    return dataframe




# simulation
'''
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

            
            stan_mc_mean, stan_mc_std = pystan_mcmc(xdata, ydata, num_points, num_points)
            stan_vi_mean, stan_vi_std = pystan_vi(xdata, ydata, num_points, num_points)

            result_dict['delta_stan_mc_mean'] = stan_mc_mean
            result_dict['delta_stan_mc_std'] = stan_mc_std
            result_dict['delta_stan_vi_mean'] = stan_vi_mean
            result_dict['delta_stan_vi_std'] = stan_vi_std
            
            
            pymc3_mc_mean, pymc3_mc_std = pymc3_mcmc(xdata, ydata)
            pymc3_vi_mean, pymc3_vi_std = pymc3_vi(xdata, ydata)

            result_dict['delta_pymc3_mc_mean'] = pymc3_mc_mean
            result_dict['delta_pymc3_mc_std'] = pymc3_mc_std
            result_dict['delta_pymc3_vi_mean'] = pymc3_vi_mean
            result_dict['delta_pymc3_vi_std'] = pymc3_vi_std
            

            edward_vi_mean, edward_vi_std = edward_vi(xdata, ydata, num_points, num_points)
            result_dict['delta_edward_vi_mean'] = edward_vi_mean
            result_dict['delta_edward_vi_std'] = edward_vi_std

            result.append(result_dict)

end_time = time.time()
print("--- total time %s seconds ---" % (end_time - start_time))
print(result)
'''

# real
start_time = time.time()

real_data = read_csv("segmented_sorting_fasion_floor_fashion_processed.csv")
real_data = addDerivedKPIColumn(real_data, "CTR", "orders", "sessions")
xdata = real_data.loc[real_data.variant == 'Control', 'CTR'].as_matrix()
ydata = real_data.loc[real_data.variant == 'Treatment', 'CTR'].as_matrix()

print("num of xdata", len(xdata))
print("num of ydata", len(ydata))

result_dict = {'true delta': xdata.mean() - ydata.mean()}

# stan_mc_mean, stan_mc_std = pystan_mcmc(xdata, ydata, len(xdata), len(ydata))
# stan_vi_mean, stan_vi_std = pystan_vi(xdata, ydata, len(xdata), len(ydata))
# result_dict['delta_stan_mc_mean'] = stan_mc_mean
# result_dict['delta_stan_mc_std'] = stan_mc_std
# result_dict['delta_stan_vi_mean'] = stan_vi_mean
# result_dict['delta_stan_vi_std'] = stan_vi_std

# pymc3_mc_mean, pymc3_mc_std = pymc3_mcmc(xdata, ydata)
# pymc3_vi_mean, pymc3_vi_std = pymc3_vi(xdata, ydata)
# result_dict['delta_pymc3_mc_mean'] = pymc3_mc_mean
# result_dict['delta_pymc3_mc_std'] = pymc3_mc_std
# result_dict['delta_pymc3_vi_mean'] = pymc3_vi_mean
# result_dict['delta_pymc3_vi_std'] = pymc3_vi_std

edward_vi_mean, edward_vi_std = edward_vi(xdata, ydata, len(xdata), len(ydata))
result_dict['delta_edward_vi_mean'] = edward_vi_mean
result_dict['delta_edward_vi_std'] = edward_vi_std

end_time = time.time()
print("--- total time %s seconds ---" % (end_time - start_time))
print(result_dict)