import matplotlib as mpl
from matplotlib import pylab as plt
import matplotlib.dates as mdates
import seaborn as sns

import collections

import numpy as np
import pandas as pd
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability import distributions as tfd
from tensorflow_probability import sts

tf.enable_v2_behavior()


from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

sns.set_context("notebook", font_scale=1.)
sns.set_style("whitegrid")



def plot_forecast(x, y,
                  forecast_mean, forecast_scale, forecast_samples,
                  title, x_locator=None, x_formatter=None):
  """Plot a forecast distribution against the 'true' time series."""
  colors = sns.color_palette()
  c1, c2 = colors[0], colors[1]
  fig = plt.figure(figsize=(12, 6))
  ax = fig.add_subplot(1, 1, 1)

  num_steps = len(y)
  num_steps_forecast = forecast_mean.shape[-1]
  num_steps_train = num_steps - num_steps_forecast


  ax.plot(x, y, lw=2, color=c1, label='ground truth')

  forecast_steps = np.arange(
      x[num_steps_train],
      x[num_steps_train]+num_steps_forecast,
      dtype=x.dtype)

  ax.plot(forecast_steps, forecast_samples.T, lw=1, color=c2, alpha=0.1)

  ax.plot(forecast_steps, forecast_mean, lw=2, ls='--', color=c2,
           label='forecast')
  ax.fill_between(forecast_steps,
                   forecast_mean-2*forecast_scale,
                   forecast_mean+2*forecast_scale, color=c2, alpha=0.2)

  ymin, ymax = min(np.min(forecast_samples), np.min(y)), max(np.max(forecast_samples), np.max(y))
  yrange = ymax-ymin
  ax.set_ylim([ymin - yrange*0.1, ymax + yrange*0.1])
  ax.set_title("{}".format(title))
  ax.legend()

  if x_locator is not None:
    ax.xaxis.set_major_locator(x_locator)
    ax.xaxis.set_major_formatter(x_formatter)
    fig.autofmt_xdate()

  return fig, ax


def plot_components(dates,
                    component_means_dict,
                    component_stddevs_dict,
                    x_locator=None,
                    x_formatter=None):
  """Plot the contributions of posterior components in a single figure."""
  colors = sns.color_palette()
  c1, c2 = colors[0], colors[1]

  axes_dict = collections.OrderedDict()
  num_components = len(component_means_dict)
  fig = plt.figure(figsize=(12, 2.5 * num_components))
  for i, component_name in enumerate(component_means_dict.keys()):
    component_mean = component_means_dict[component_name]
    component_stddev = component_stddevs_dict[component_name]

    ax = fig.add_subplot(num_components,1,1+i)
    ax.plot(dates, component_mean, lw=2)
    ax.fill_between(dates,
                     component_mean-2*component_stddev,
                     component_mean+2*component_stddev,
                     color=c2, alpha=0.5)
    ax.set_title(component_name)
    if x_locator is not None:
      ax.xaxis.set_major_locator(x_locator)
      ax.xaxis.set_major_formatter(x_formatter)
    axes_dict[component_name] = ax
  fig.autofmt_xdate()
  fig.tight_layout()
  return fig, axes_dict


def plot_one_step_predictive(dates, observed_time_series,
                             one_step_mean, one_step_scale,
                             x_locator=None, x_formatter=None):
  """Plot a time series against a model's one-step predictions."""

  colors = sns.color_palette()
  c1, c2 = colors[0], colors[1]

  fig=plt.figure(figsize=(12, 6))
  ax = fig.add_subplot(1,1,1)
  num_timesteps = one_step_mean.shape[-1]
  ax.plot(dates, observed_time_series, label="observed time series", color=c1)
  ax.plot(dates, one_step_mean, label="one-step prediction", color=c2)
  ax.fill_between(dates,
                  one_step_mean - one_step_scale,
                  one_step_mean + one_step_scale,
                  alpha=0.1, color=c2)
  ax.legend()

  if x_locator is not None:
    ax.xaxis.set_major_locator(x_locator)
    ax.xaxis.set_major_formatter(x_formatter)
    fig.autofmt_xdate()
  fig.tight_layout()
  return fig, ax



# Time Series Data of San Francisco Crime report, from 2018 to 2023
# Original source: https://data.sfgov.org/Public-Safety/Police-Department-Incident-Reports-2018-to-Present/wg3w-h783
# San Francisco Police Department
# And Covid Occurance data from 2021 to 2022
# dataset was incident-based data; here we will downsample it to time data

crime_count = pd.read_csv("Sanfrancisco_crime_report_full.csv", delimiter=",")
crime_count['Incident Time'] = crime_count['Incident Time'].str[:2]
crime_count['count'] = 1
crime_count = crime_count[['Incident Date', 'Incident Time', 'count']].groupby(['Incident Date', 'Incident Time']).sum().reset_index()
crime_count = crime_count['count'].values.astype(np.float32)

crime_count_dates = np.arange('2018-01-01', '2023-08-08', dtype='datetime64[h]')
crime_count_loc = mdates.WeekdayLocator(byweekday=mdates.WE)
crime_count_fmt = mdates.DateFormatter('%a %y %b %d')


covid_count = pd.read_csv("Sanfrancisco_covid_count.csv", delimiter=",")
covid_count = covid_count['new_cases'].values.astype(np.float32)


# covid data starts at 21/01/01 while police data starts at 18/01/01, so we fill the empty period
covid_case_pre_fill = np.zeros((365*3,), dtype=np.float32)
covid_case = np.repeat(covid_count, 24)
covid_case = np.append(covid_case_pre_fill, covid_case)
len_diff = crime_count.__len__() - covid_case.__len__()
covid_case_end_fill = np.zeros((len_diff,), dtype=np.float32)
covid_case = np.append(covid_case, covid_case_end_fill)



num_forecast_steps = 24 * 7 * 4 * 6 # Six Month.
crime_count_training_data = crime_count[:-num_forecast_steps]

colors = sns.color_palette()
c1, c2 = colors[0], colors[1]

fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(2, 1, 1)
ax.plot(crime_count_dates[:-num_forecast_steps],
        crime_count[:-num_forecast_steps], lw=2, label="training data")
ax.set_ylabel("Hourly demand (GW)")



ax = fig.add_subplot(2, 1, 2)

ax.plot(crime_count_dates[:-num_forecast_steps],
        covid_case[:-num_forecast_steps], lw=2, label="training data", c=c2)
ax.set_ylabel("Covid Case")
ax.set_title("Covid Case")
ax.xaxis.set_major_locator(crime_count_loc)
ax.xaxis.set_major_formatter(crime_count_fmt)
fig.suptitle("Covid Patient Occurance in San Francisco, USA",
             fontsize=15)
fig.autofmt_xdate()



def build_model(observed_time_series):
  hour_of_day_effect = sts.Seasonal(
      num_seasons=24,
      observed_time_series=observed_time_series,
      name='hour_of_day_effect')
  day_of_week_effect = sts.Seasonal(
      num_seasons=7, num_steps_per_season=24,
      observed_time_series=observed_time_series,
      name='day_of_week_effect')
  temperature_effect = sts.LinearRegression(
      design_matrix=tf.reshape(temperature - np.mean(temperature),
                               (-1, 1)), name='temperature_effect')
  autoregressive = sts.Autoregressive(
      order=1,
      observed_time_series=observed_time_series,
      name='autoregressive')
  model = sts.Sum([hour_of_day_effect,
                   day_of_week_effect,
                   temperature_effect,
                   autoregressive],
                   observed_time_series=observed_time_series)
  return model


demand_model = build_model(demand_training_data)

# Build the variational surrogate posteriors `qs`.
variational_posteriors = tfp.sts.build_factored_surrogate_posterior(
    model=demand_model)


#@title Minimize the variational loss.

# Allow external control of optimization to reduce test runtimes.
num_variational_steps = 200 # @param { isTemplate: true}
num_variational_steps = int(num_variational_steps)

# Build and optimize the variational loss function.
elbo_loss_curve = tfp.vi.fit_surrogate_posterior(
    target_log_prob_fn=demand_model.joint_distribution(
        observed_time_series=demand_training_data).log_prob,
    surrogate_posterior=variational_posteriors,
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    num_steps=num_variational_steps,
    jit_compile=True)
plt.plot(elbo_loss_curve)
plt.show()

# Draw samples from the variational posterior.
q_samples_demand_ = variational_posteriors.sample(50)



print("Inferred parameters:")
for param in demand_model.parameters:
  print("{}: {} +- {}".format(param.name,
                              np.mean(q_samples_demand_[param.name], axis=0),
                              np.std(q_samples_demand_[param.name], axis=0)))



demand_forecast_dist = tfp.sts.forecast(
    model=demand_model,
    observed_time_series=demand_training_data,
    parameter_samples=q_samples_demand_,
    num_steps_forecast=num_forecast_steps)


num_samples=10

(demand_forecast_mean, demand_forecast_scale, demand_forecast_samples) = (
    demand_forecast_dist.mean().numpy()[..., 0], demand_forecast_dist.stddev().numpy()[..., 0],
    demand_forecast_dist.sample(num_samples).numpy()[..., 0])


fig, ax = plot_forecast(demand_dates, demand,
                        demand_forecast_mean,
                        demand_forecast_scale,
                        demand_forecast_samples,
                        title="Electricity demand forecast",
                        x_locator=demand_loc, x_formatter=demand_fmt)
ax.set_ylim([0, 10])
fig.tight_layout()


# Get the distributions over component outputs from the posterior marginals on
# training data, and from the forecast model.
component_dists = sts.decompose_by_component(
    demand_model,
    observed_time_series=demand_training_data,
    parameter_samples=q_samples_demand_)

forecast_component_dists = sts.decompose_forecast_by_component(
    demand_model,
    forecast_dist=demand_forecast_dist,
    parameter_samples=q_samples_demand_)


demand_component_means_, demand_component_stddevs_ = (
    {k.name: c.mean() for k, c in component_dists.items()},
    {k.name: c.stddev() for k, c in component_dists.items()})

(
    demand_forecast_component_means_,
    demand_forecast_component_stddevs_
) = (
    {k.name: c.mean() for k, c in forecast_component_dists.items()},
    {k.name: c.stddev() for k, c in forecast_component_dists.items()}
    )

# Concatenate the training data with forecasts for plotting.
component_with_forecast_means_ = collections.OrderedDict()
component_with_forecast_stddevs_ = collections.OrderedDict()
for k in demand_component_means_.keys():
  component_with_forecast_means_[k] = np.concatenate([
      demand_component_means_[k],
      demand_forecast_component_means_[k]], axis=-1)
  component_with_forecast_stddevs_[k] = np.concatenate([
      demand_component_stddevs_[k],
      demand_forecast_component_stddevs_[k]], axis=-1)


fig, axes = plot_components(
  demand_dates,
  component_with_forecast_means_,
  component_with_forecast_stddevs_,
  x_locator=demand_loc, x_formatter=demand_fmt)
for ax in axes.values():
  ax.axvline(demand_dates[-num_forecast_steps], linestyle="--", color='red')


demand_one_step_dist = sts.one_step_predictive(
    demand_model,
    observed_time_series=demand,
    parameter_samples=q_samples_demand_)

demand_one_step_mean, demand_one_step_scale = (
    demand_one_step_dist.mean().numpy(), demand_one_step_dist.stddev().numpy())



fig, ax = plot_one_step_predictive(
    demand_dates, demand,
    demand_one_step_mean, demand_one_step_scale,
    x_locator=demand_loc, x_formatter=demand_fmt)
ax.set_ylim(0, 10)

# Use the one-step-ahead forecasts to detect anomalous timesteps.
zscores = np.abs((demand - demand_one_step_mean) /
                 demand_one_step_scale)
anomalies = zscores > 3.0
ax.scatter(demand_dates[anomalies],
           demand[anomalies],
           c="red", marker="x", s=20, linewidth=2, label=r"Anomalies (>3$\sigma$)")
ax.plot(demand_dates, zscores, color="black", alpha=0.1, label='predictive z-score')
ax.legend()
plt.show()
