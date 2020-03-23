import numpy as np
import pandas as pd
from urllib.parse import urljoin


# Melt the pd.DataFrame into the right shape and set index
def clean_data(df_raw):
    df_cleaned=df_raw.melt(
        id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'],
        value_name='Cases',
        var_name='Date'
    )
    df_cleaned = df_cleaned.set_index(['Country/Region', 'Province/State', 'Date'])
    return df_cleaned


# Get data per country
def get_country_data(df_cleaned, old_name, new_name):
    df_country = df_cleaned.groupby(['Country/Region', 'Date'])['Cases'].sum().reset_index()
    df_country = df_country.set_index(['Country/Region', 'Date'])
    df_country.index = df_country.index.set_levels(
        [df_country.index.levels[0], pd.to_datetime(df_country.index.levels[1])])
    df_country = df_country.sort_values(['Country/Region', 'Date'], ascending=True)
    df_country = df_country.rename(columns={old_name: new_name})
    return df_country


# Get DailyData from Cumulative sum
def get_daily_data(df_country, old_name, new_name):
    df_country_daily = df_country.groupby(level=0).diff().fillna(0)
    df_country_daily = df_country_daily.rename(columns={old_name: new_name})
    return df_country_daily


def get_corona_data():
    link = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/'

    confirmed_cases_raw = pd.read_csv(link + 'time_series_19-covid-Confirmed.csv')
    deaths_raw = pd.read_csv(link + 'time_series_19-covid-Deaths.csv')
    recoveries_raw = pd.read_csv(link + 'time_series_19-covid-Recovered.csv')

    # Clean all pd.DataFrames
    confirmed_cases = clean_data(confirmed_cases_raw)
    deaths = clean_data(deaths_raw)
    recoveries = clean_data(recoveries_raw)

    confirmed_cases_country = get_country_data(confirmed_cases, 'Cases', 'Total Confirmed Cases')
    deaths_country = get_country_data(deaths, 'Cases', 'Total Deaths')
    recoveries_country = get_country_data(recoveries, 'Cases', 'Total Recoveries')

    new_cases_country = get_daily_data(confirmed_cases_country, 'Total Confirmed Cases', 'Daily New Cases')
    new_deaths_country = get_daily_data(deaths_country, 'Total Deaths', 'Daily New Deaths')
    new_recoveries_country = get_daily_data(recoveries_country, 'Total Recoveries', 'Daily New Recoveries')

    corona_data = pd.merge(confirmed_cases_country, new_cases_country, how='left', left_index=True, right_index=True)
    corona_data = pd.merge(corona_data, new_deaths_country, how='left', left_index=True, right_index=True)
    corona_data = pd.merge(corona_data, deaths_country, how='left', left_index=True, right_index=True)
    corona_data = pd.merge(corona_data, recoveries_country, how='left', left_index=True, right_index=True)
    corona_data = pd.merge(corona_data, new_recoveries_country, how='left', left_index=True, right_index=True)
    corona_data['Active Cases'] = corona_data['Total Confirmed Cases'] - corona_data['Total Deaths'] - corona_data[
        'Total Recoveries']
    corona_data['Share of Recoveries - Closed Cases'] = np.round(
        corona_data['Total Recoveries'] / (corona_data['Total Recoveries'] + corona_data['Total Deaths']), 2)
    corona_data['Death to Cases Ratio'] = np.round(corona_data['Total Deaths'] / corona_data['Total Confirmed Cases'],
                                                   3)

    return corona_data


if __name__ == '__main__':
    corona_data = get_corona_data()