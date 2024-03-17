import pandas as pd
import numpy as np
import seaborn as sns
import geopandas as gpd
import matplotlib.pyplot as plt
from typing import Optional, Union

from transforms import Transform, Composite
from data_processor import DataProcessor


def load_commodities_dataframes(
    data_frame: pd.DataFrame,
    input_encoder: Optional[Union[Transform, Composite]] = None,
    output_encoder: Optional[Union[Transform, Composite]] = None,
    verbose: bool = False,
) -> tuple:
    """Load commodities dataframes, and DataProcessor.

    Returns both to recover operations down pipeline.

    Args
    ----
        :param data_frame: commodities dataframe
        :param input_encoder: input encoder
        :param output_encoder: output encoder
        :param verbose: human logging

    Returns
    -------
        :return: processed dataframe, data_processor
    """

    data_processor = DataProcessor(
        input_encoder=input_encoder, output_encoder=output_encoder
    )

    if input_encoder:
        data_frame = data_processor.preprocess(data_frame)

    if output_encoder:
        data_frame = output_encoder.postprocess(data_frame)

    return data_frame, data_processor


def plot_commodities_atemporal_ppu_distribution(
    commodity: str,
):
    """Rough histogram of atemporal commodity unitary prices"""

    data_frame = pd.read_csv(f"data/clean/{commodity}_normalised.csv")

    plt.figure(figsize=(10, 6))
    plt.hist(data_frame["unit_price"], bins=50, edgecolor="black")
    plt.title("Distribution of Unitary Prices")
    plt.xlabel("Unitary Price")
    plt.ylabel("Frequency")

    # Plot a density plot
    sns.kdeplot(data_frame["unit_price"], bw_method=0.3)
    plt.title("Density Plot of Unitary Prices")
    plt.xlabel("Unitary Price")
    plt.ylabel("Density")
    plt.show()


def plot_commodities_temporal_ppu_distribution(commodity: str):
    """Rough histogram of averaged monthly unitary price distribution"""

    data_frame = pd.read_csv(f"data/clean/{commodity}_normalised.csv")

    data_frame["date"] = pd.to_datetime(data_frame[["year", "month"]].assign(day=1))

    # Group by year and month and calculate the mean unit price
    grouped = (
        data_frame.groupby(pd.Grouper(key="date", freq="M"))["unit_price"]
        .mean()
        .reset_index()
    )

    # Plot a histogram of the mean unit prices
    plt.figure(figsize=(10, 6))
    plt.hist(grouped["unit_price"], bins=50, edgecolor="black")
    plt.title("Distribution of Average Monthly Unitary Prices")
    plt.xlabel("Average Monthly Unitary Price")
    plt.ylabel("Frequency")

    # Plot a density plot of the mean unit prices
    sns.kdeplot(grouped["unit_price"], bw_method=0.3)
    plt.title("Density Plot of Average Monthly Unitary Prices")
    plt.xlabel("Average Monthly Unitary Price")
    plt.ylabel("Density")
    plt.show()


def plot_commodities_atemporal_ppu_cumulative_distribution(commodity: str):
    """Cumulative distribution of atemporal commodity unitary prices"""

    data_frame = pd.read_csv(f"data/clean/{commodity}_normalised.csv")

    plt.figure(figsize=(10, 6))
    plt.hist(data_frame["unit_price"], bins=50, edgecolor="black", cumulative=True)
    plt.title("Cumulative Distribution of Unitary Prices")
    plt.xlabel("Unitary Price")
    plt.ylabel("Cumulative Frequency")
    plt.show()


def plot_commodities_temporal_ppu_cumulative_distribution(commodity: str):
    """Cumulative distribution of averaged monthly unitary price"""

    data_frame = pd.read_csv(f"data/clean/{commodity}_normalised.csv")

    data_frame["date"] = pd.to_datetime(data_frame[["year", "month"]].assign(day=1))

    # Group by year and month and calculate the mean unit price
    grouped = (
        data_frame.groupby(pd.Grouper(key="date", freq="M"))["unit_price"]
        .mean()
        .reset_index()
    )

    # Plot a histogram of the mean unit prices
    plt.figure(figsize=(10, 6))
    plt.hist(grouped["unit_price"], bins=50, edgecolor="black", cumulative=True)
    plt.title("Cumulative Distribution of Average Monthly Unitary Prices")
    plt.xlabel("Average Monthly Unitary Price")
    plt.ylabel("Cumulative Frequency")
    plt.show()


def plot_commodity_geotemporal_export_gradients(
    commodity: str,
):
    """Plots geographically sectioned, monthly averaged export gradients on country scale"""

    data_frame = pd.read_csv(f"data/clean/{commodity}_normalised.csv")

    # Convert the 'year' and 'month' columns to datetime
    data_frame["date"] = pd.to_datetime(data_frame[["year", "month"]].assign(day=1))

    # Group by export country and date, and calculate the average monthly export quantity
    grouped = (
        data_frame.groupby(["export_country_code", pd.Grouper(key="date", freq="M")])[
            "quantity"
        ]
        .mean()
        .reset_index()
    )

    # Calculate the overall average monthly export quantity for each country
    country_export_quantity = grouped.groupby("export_country_code")["quantity"].mean()

    # Load the world GeoDataFrame
    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))

    # Merge the world GeoDataFrame with the country export quantity data
    merged = world.set_index("iso_a3").join(country_export_quantity, how="left")

    # Plot the world map with the average monthly export quantity used to determine the color of each country
    fig, ax = plt.subplots(1, 1)
    merged.plot(
        column="quantity", ax=ax, legend=True, missing_kwds={"color": "lightgrey"}
    )
    plt.show()


def plot_commodity_geotemporal_import_gradients(
    commodity: str,
):
    """Plots geographically sectioned, monthly averaged import gradients on country scale"""

    data_frame = pd.read_csv(f"data/clean/{commodity}_normalised.csv")

    # Convert the 'year' and 'month' columns to datetime
    data_frame["date"] = pd.to_datetime(data_frame[["year", "month"]].assign(day=1))

    # Group by import country and date, and calculate the average monthly import quantity
    grouped = (
        data_frame.groupby(["import_country_code", pd.Grouper(key="date", freq="M")])[
            "quantity"
        ]
        .mean()
        .reset_index()
    )

    # Calculate the overall average monthly import quantity for each country
    country_import_quantity = grouped.groupby("import_country_code")["quantity"].mean()

    # Load the world GeoDataFrame
    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))

    # Merge the world GeoDataFrame with the country import quantity data
    merged = world.set_index("iso_a3").join(country_import_quantity, how="left")

    # Plot the world map with the average monthly import quantity used to determine the color of each country
    fig, ax = plt.subplots(1, 1)
    merged.plot(
        column="quantity", ax=ax, legend=True, missing_kwds={"color": "lightgrey"}
    )
    plt.show()


def qq_commodity_unitary_price_plot(
    commodity: str,
):
    """Q-Q plot of commodity unitary prices"""

    data_frame = pd.read_csv(f"data/clean/{commodity}_normalised.csv")
