import abc
import numpy as np
import pandas as pd

from typing import List, Union


class Transform(abc.ABC):
    """Transform base class"""

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def transform(self, data_frame: pd.DataFrame):
        """Forward transform"""
        pass

    @abc.abstractmethod
    def inverse_transform(self, data_frame: pd.DataFrame):
        """Inverse transform"""
        pass


class Composite(Transform):
    """Composite transform for multiprocess data transforms

    Input transformations must be instantiations of the
    Transform class.

    Args
    ----
        :param transforms: transformations pipeline
    """

    def __init__(self, transforms: List[Transform]):
        super().__init__()
        self.transforms = transforms

    def transform(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        """Forward transfroms"""
        for transform in self.transforms:
            data_frame = transform.transform(data_frame)

        return data_frame

    def inverse_transform(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        """Inverse transforms"""
        for transform in reversed(self.transforms):
            data_frame = transform.inverse_transform(data_frame)

        return data_frame


class AppendUnitaryPrice(Transform):
    """Engineers unitary price of commodity dataset"""

    def __init__(self):
        super().__init__()
        self.field = "unit_price"

    def transform(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        """Forward transform: total price / quantity -> unitary price"""
        unit_price = np.array(
            data_frame["value"].values / data_frame["quantity"].values
        )
        unit_price[np.isinf(unit_price)] = 0
        unit_price[np.isnan(unit_price)] = 0
        data_frame[self.field] = unit_price
        return data_frame

    def inverse_transform(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        """Inverse transform: drop column"""
        data_frame = data_frame.drop(columns=[self.field])
        return data_frame


class GeographicFilter(Transform):
    """Filters and formats import/export countries"""

    def __init__(self):
        super().__init__()
        self.fields = [
            "export_country",
            "import_country",
        ]

    def transform(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        """Forward transform"""
        data_frame = data_frame[
            ~(
                data_frame[self.fields[0]].str.contains(".nes")
                | data_frame[self.fields[1]].str.contains(".nes")
            )
        ]

        data_frame[self.fields[0]] = [
            item[0]
            for item in np.char.split(
                data_frame[self.fields[0]].values.astype(str), " (mainland)"
            )
        ]

        data_frame[self.fields[1]] = [
            item[0]
            for item in np.char.split(
                data_frame[self.fields[1]].values.astype(str), " (mainland)"
            )
        ]
        return data_frame

    def inverse_transform(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        """Inverse transform obviated, no-op"""
        return data_frame


class GeographicEnhancer(Transform):
    """Augments geographic information on import/export countries
    at different scales wrt different criteria for analysis.
    """

    def __init__(self):
        super().__init__()
        countries: pd.DataFrame = pd.read_csv("data/clean/countries.csv")
        self.countries = countries.iloc[1:, :]
        self.countries.reset_index(drop=True, inplace=True)
        self.fields = [
            "export_country",
            "import_country",
        ]

    def transform(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        """Forward transform adding latitude, longitude,
        region, subregion, ISO Alpha-3 code"""

        data_frame = data_frame[
            np.isin(
                data_frame[self.fields[0]].values,
                self.countries["Preferred Term"].values,
            )
            | np.isin(
                data_frame[self.fields[0]].values,
                self.countries["ISO Alt Term"].values,
            )
            | np.isin(
                data_frame[self.fields[0]].values,
                self.countries["English Short"].values,
            )
            | np.isin(
                data_frame[self.fields[0]].values,
                self.countries["FTS Alt Term"].values,
            )
        ]

        data_frame = data_frame[
            np.isin(
                data_frame[self.fields[1]].values,
                self.countries["Preferred Term"].values,
            )
            | np.isin(
                data_frame[self.fields[1]].values,
                self.countries["ISO Alt Term"].values,
            )
            | np.isin(
                data_frame[self.fields[1]].values,
                self.countries["English Short"].values,
            )
            | np.isin(
                data_frame[self.fields[1]].values,
                self.countries["FTS Alt Term"].values,
            )
        ]

        for field in self.fields:
            for country in data_frame[field].unique():
                j = np.where(
                    (self.countries["Preferred Term"] == country)
                    | (self.countries["m49 Alt Term"] == country)
                    | (self.countries["ISO Alt Term"] == country)
                    | (self.countries["English Short"] == country)
                    | (self.countries["FTS Alt Term"] == country)
                    | (self.countries["HRinfo Alt Term"] == country)
                )[0][0]
                data_frame.loc[data_frame[field] == country, field + "_region"] = (
                    self.countries.loc[j, "Region Name"]
                )
                data_frame.loc[data_frame[field] == country, field + "_subregion"] = (
                    self.countries.loc[j, "Sub-region Name"]
                )
                data_frame.loc[data_frame[field] == country, field + "_code"] = (
                    self.countries.loc[j, "ISO 3166-1 Alpha 3-Codes"]
                )
                data_frame.loc[data_frame[field] == country, field + "_latitude"] = (
                    self.countries.loc[j, "Latitude"]
                )
                data_frame.loc[data_frame[field] == country, field + "_longitude"] = (
                    self.countries.loc[j, "Longitude"]
                )

        data_frame = data_frame[
            [
                "year",
                "month",
                "export_country",
                "export_country_code",
                "export_country_region",
                "export_country_subregion",
                "export_country_latitude",
                "export_country_longitude",
                "import_country",
                "import_country_code",
                "import_country_region",
                "import_country_subregion",
                "import_country_latitude",
                "import_country_longitude",
                "commodity",
                "value",
                "quantity",
                "unit_price",
            ]
        ]

        data_frame.dropna(inplace=True)
        data_frame.reset_index(drop=True, inplace=True)

        return data_frame

    def inverse_transform(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        """Inverse transform obviated, no-op"""
        return data_frame


class UnitGaussianNormalizer(Transform):
    """Normalizes data to unit Gaussian

    Args
    ----
        :param dims: dimensional discriminator, specifies
                     columns along which to normalize; in
                     this case since DataFrame, strings or
                     list of strings
        :param save_checkpoint: save flag, storing for plots
    """

    def __init__(self, dims: Union[str, List[str]], save_checkpoint: bool = False):
        super().__init__()
        if isinstance(dims, str):
            dims = [dims]
        self.dims: list = dims
        self.save_checkpoint: bool = save_checkpoint

    def transform(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        """Forward transform"""
        for dim in self.dims:
            values: np.array = data_frame[dim].values
            data_frame[dim] = (values - np.mean(values)) / np.std(values)

        if self.save_checkpoint:
            save_name = data_frame["commodity"].iloc[0]
            data_frame.to_csv(f"data/clean/{save_name}_normalised.csv", index=False)
        return data_frame

    def inverse_transform(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        """Inverse transform"""
        for dim in self.dims:
            values: np.array = data_frame[dim].values
            data_frame[dim] = values * np.std(values) + np.mean(values)
        return data_frame


class NatLogScaler(Transform):
    """Scales data to natural logarithm

    Args
    ----
    :param dims: dimensional discriminator, specifies
                 columns along which to normalize; in
                 this case since DataFrame, strings or
                 list of strings"""

    def __init__(self, dims: Union[str, List[str]]):
        super().__init__()
        if isinstance(dims, str):
            dims = [dims]
        self.dims: list = dims

    def transform(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        """Forward transform"""
        for dim in self.dims:
            data_frame[dim] = np.log(data_frame[dim].values)
        return data_frame

    def inverse_transform(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        """Inverse transform"""
        for dim in self.dims:
            data_frame[dim] = np.exp(data_frame[dim].values)
        return data_frame


class StatisticalSignificanceFilter(Transform):
    """Proxies statistical significance based on proportional
    contribution of commodity trades wrt aggragate volume, and
    drops ouliers that are not representative due to extreme"""

    # TODO: yet implemented in pipeline; first must analyse
    # fat tails on distribution of unitary price?????
    # how do I estimate the CCDF?
    # How can I ascertain statistical insignificance of data
