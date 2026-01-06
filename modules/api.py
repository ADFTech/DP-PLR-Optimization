import requests
from datetime import datetime, time
from modules.models.vc_weather_model import VC_weather_model
from modules.models.accu_weather_model import Accu_weather_model

class Weather_api():
    """
    A class for querying the Visual Crossing Weather API and AccuWeather API.

    Class attributes:
        FORECAST_API_ENDPOINT (str): The endpoint for querying the Accu Weather API for a weather forecast.
        forecast__ENDPOINT (str): The endpoint for querying the Visual Crossing Weather API for historical weather data.

    Attributes:
        forecast_params (dict): The parameters to pass to the forecast API.
        history_params (dict): The parameters to pass to the history API.
    """

    FORECAST_API_ENDPOINT = "https://dataservice.accuweather.com/forecasts/v1/hourly"

    HISTORY_API_ENDPOINT = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/weatherdata/history"

    def __init__(self, config: dict, forecast_params: dict, history_params: dict) -> None:
        self.config = config
        self.forecast_params = forecast_params
        self.history_params = history_params

    @property
    def config(self) -> dict:
        """
        Gets the config.
        """
        return self.__config

    @config.setter
    def config(self, config: dict) -> None:
        """
        Sets the config.
        """
        if not isinstance(config, dict):
            raise ValueError("Config must be a dictionary")
        self.__config = config

    @property
    def forecast_params(self) -> dict:
        """
        Gets the forecast params.
        """
        return self.__forecast_params

    @forecast_params.setter
    def forecast_params(self, params: dict) -> None:
        """
        Sets the forecast params.
        """
        if not isinstance(params, dict):
            raise ValueError("Forecast params must be a dictionary")
        self.__forecast_params = params

    @property
    def history_params(self) -> dict:
        """
        Gets the history params.
        """
        return self.__history_params

    @history_params.setter
    def history_params(self, params: dict) -> None:
        """
        Sets the history params.
        """
        if not isinstance(params, dict):
            raise ValueError("History params must be a dictionary")
        self.__history_params = params

    def __query(self, endpoint: str, params: dict) -> dict:
        """
        A private (generic) method for querying the Visual Crossing Weather API.

        Args:
            endpoint (str): The endpoint to query.
            params (dict): The parameters to pass to the API.

        Returns:
            dict: A dictionary response.
        """
        if not isinstance(endpoint, str):
            raise ValueError("Endpoint must be a string")

        if not isinstance(params, dict):
            raise ValueError("Params must be a dictionary")

        response = requests.get(endpoint, params)
        if response.status_code != 200:
            raise Exception(response.text)

        return response.json()

    def forecast(self, **kwargs: dict) -> Accu_weather_model:
        """
        A method for querying the Accu Weather API for a weather forecast.
        It returns the forecasted weather data for the next 12 hours.

        It keeps a record of all the weather data it has queried.

        Args:
            **kwargs (dict): Additional parameters to pass to the API.

        Returns:
            Weather_model: A Weather_model object.
        """
        return Accu_weather_model(
            weather_data = self.__query(
                endpoint = f"{self.FORECAST_API_ENDPOINT}/{self.config['forecast_window']}/{self.config['forecast_location']}",
                params = {
                    **self.forecast_params,
                    **kwargs,
                }
            )
        )

    def history(self, date: datetime|tuple[datetime, datetime], **kwargs: dict) -> VC_weather_model:
        """
        A class method for querying the Visual Crossing Weather API for historical weather data.
        It takes a location and a date or tuple of dates and returns a VC_weather_model object.

        It keeps a record of all the weather data it has queried.

        Args:
            date (datetime|tuple[datetime, datetime]): The date or tuple of dates to query.
            **kwargs (dict): Additional parameters to pass to the API.

        Returns:
            Weather_model: A Weather_model object.
        """


        if isinstance(date, datetime):
            date = (
                datetime.combine(date.date(), time.min).strftime('%Y-%m-%dT%H:%M:%S'),
                datetime.combine(date.date(), time.max).strftime('%Y-%m-%dT%H:%M:%S')
            )
        elif isinstance(date, tuple):
            if len(date) != 2 or not all(isinstance(d, datetime) for d in date):
                raise ValueError("Date must be a tuple of two datetime objects")

            # If the dates are in the right order, use them as-is
            if date[0] < date[1]:
                # Create a tuple of the start and end of the day. formatted as a string
                date = (
                    datetime.combine(date[0].date(), time.min).strftime('%Y-%m-%dT%H:%M:%S'),
                    datetime.combine(date[1].date(), time.max).strftime('%Y-%m-%dT%H:%M:%S')
                )
            # If the dates are in the wrong order, swap them
            else:
                # Create a tuple of the start and end of the day. formatted as a string
                date = (
                    datetime.combine(date[1].date(), time.min).strftime('%Y-%m-%dT%H:%M:%S'),
                    datetime.combine(date[0].date(), time.max).strftime('%Y-%m-%dT%H:%M:%S')
                )
        else:
            raise ValueError("Date must be a datetime object or a tuple of two datetime objects")

        start_time, end_time = date

        return VC_weather_model(
            **self.__query(
                endpoint = f"{self.HISTORY_API_ENDPOINT}",
                params = {
                    'locations': self.config['historical_location'],
                    'startDateTime': start_time,
                    'endDateTime': end_time,
                    **self.history_params,
                    **kwargs,
                }
            )
        )
