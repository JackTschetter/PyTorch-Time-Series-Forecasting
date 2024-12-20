from multi_prophet.prophet import Prophet
from multi_prophet.factories import model_pool_factory, dataframe_builder_factory
import multiprocessing as mp
import pandas as pd
import concurrent.futures
from concurrent.futures import wait
from utils_plot import suppress_stdout_stderr
from multiprocessing import Pool
import time
import tqdm

__version__ = "1.1.1"


def mp_fit_task(input):
    column, mdf, model = input
    with suppress_stdout_stderr():
        model.fit(mdf)
    return column, model


class MultiProphet:
    def __init__(self, columns=[], config=None, regressors={}, **kwargs):
        self.model_pool = model_pool_factory(columns=columns,
                                             config=config,
                                             regressors=regressors,
                                             **kwargs)
        self.df_builder = dataframe_builder_factory(regressors)

    def mp_fit(self, df, **kwargs):
        # Multiprocess implementation
        inputs = []
        for i, (column, model) in enumerate(self.model_pool.items()):
            mdf = self._create_dataframe(df, column, train=True)
            inputs.append((column, mdf, model))

        p = Pool(48)
        p_outs = list(tqdm.tqdm(p.imap(mp_fit_task, inputs), total=len(inputs)))
        p.close()
        p.join()

        for p_out in p_outs:
            column, model = p_out
            self.model_pool[column] = model

    def fit(self, df, **kwargs):
        # Default implementation
        for column, model in tqdm.tqdm(self.model_pool.items()):
            mdf = self._create_dataframe(df, column, train=True)
            with suppress_stdout_stderr():
                model.fit(mdf, **kwargs)

    def make_future_dataframe(self, periods, **kwargs):
        model = self._first_model()
        return model.make_future_dataframe(periods, **kwargs)

    def predict(self, future_df):
        return {
            column: model.predict(self._create_dataframe(future_df, column))
            for column, model in tqdm.tqdm(self.model_pool.items())
        }

    def add_seasonality(self, columns=None, **kwargs):
        columns = self._columns(columns)

        for model in self._models(columns):
            model.add_seasonality(**kwargs)

    def add_country_holidays(self, country_name, columns=None):
        columns = self._columns(columns)

        for model in self._models(columns):
            model.add_country_holidays(country_name)

    def add_regressor(self, name, columns=None, **kwargs):
        columns = self._columns(columns)

        for model in self._models(columns):
            model.add_regressor(name, **kwargs)

        self._add_regressor_to_builder(name, columns)

    def plot(self, forecasts, plotly=False, **kwargs):
        return {
            column: self.model_pool[column].plot(forecast, plotly=plotly, **kwargs)
            for column, forecast in forecasts.items()
        }

    def plot_components(self, forecasts, plotly=False, **kwargs):
        return {
            column: self.model_pool[column].plot_components(forecast,
                                                            plotly=plotly,
                                                            **kwargs)
            for column, forecast in forecasts.items()
        }

    def cross_validation(self, horizon, **kwargs):
        return {
            column: self._cross_validation(model, horizon, **kwargs)
            # column: model.cross_validation(horizon=horizon, **kwargs)
            for column, model in tqdm.tqdm(self.model_pool.items())  # Change tqdm and suppress_stdout_stderr
        }

    def _cross_validation(self, model, horizon, **kwargs):
        with suppress_stdout_stderr():
            return model.cross_validation(horizon=horizon, **kwargs)

    def performance_metrics(self, horizon, **kwargs):
        return {
            column: model.performance_metrics(horizon=horizon, **kwargs)
            for column, model in self.model_pool.items()
        }

    def _init_model_pool(self, columns, **kwargs):
        return {c: Prophet(**kwargs) for c in columns}

    def _first_model(self):
        return list(self.model_pool.values())[0]

    def _create_dataframe(self, df, column, train=False):
        return self.df_builder.create_df(df, column, train=train)

    def _contains_columns(self, df, column):
        return column in df.columns

    def _add_regressor_to_builder(self, name, columns):
        self.df_builder.add_regressor(name, columns)

    def _models(self, columns):
        return [self.model_pool[c] for c in columns]

    def _columns(self, columns):
        if columns:
            return columns
        else:
            return self.model_pool.keys()