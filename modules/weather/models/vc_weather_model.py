import math
from datetime import datetime as dt
from pydantic import BaseModel, field_validator

class VC_Weather_data(BaseModel):
    """
    A model representing detailed weather data for a specific time period.

    Attributes:
    """
    wdir: float | None = None
    uvindex: float | None = None
    preciptype: str | None = None
    cin: float | None = None
    cloudcover: float | None = None
    pop: float | None = None
    mint: float | None = None
    datetime: int | str | dt | None = None
    precip: float | None = None
    solarradiation: float | None = None
    dew: float | None = None
    humidity: float | None = None
    temp: float | None = None
    maxt: float | None = None
    visibility: float | None = None
    wspd: float | None = None
    severerisk: float | None = None
    solarenergy: float | None = None
    heatindex: float | None = None
    snowdepth: float | None = None
    sealevelpressure: float | None = None
    snow: float | None = None
    wgust: float | None = None
    conditions: str | None = None
    windchill: float | None = None
    cape: float | None = None
    sunrise: str | None = None
    icon: str | None = None
    moonphase: float | None = None
    sunset: str | None = None

    @field_validator('datetime', mode = 'after')
    def parse_datetime(cls, date):
        if isinstance(date, int):
            # Convert Unix timestamp to datetime(dt)
            length = len(str(date))
            if length == 10:  # seconds
                return dt.utcfromtimestamp(date)
            if length == 13:  # milliseconds
                return dt.utcfromtimestamp(date / 1_000.0)
            if length == 16:  # microseconds
                return dt.utcfromtimestamp(date / 1_000_000.0)
            if length == 19:  # nanoseconds
                return dt.utcfromtimestamp(date / 1_000_000_000.0)

            raise ValueError("Invalid Unix timestamp length")

        if isinstance(date, str):
            return dt.fromisoformat(date)

        return date

    @property
    def stull_wetbulb_approx(self) -> float | None:
        """
        Returns the wetbulb temperature approximation using the Stull formula.
        More info: https://www.omnicalculator.com/physics/wet-bulb
        """
        if self.temp is None or self.humidity is None:
            return None

        return \
            self.temp * math.atan(0.151977 * math.sqrt(self.humidity + 8.313659)) + \
            math.atan(self.temp + self.humidity) - math.atan(self.humidity - 1.676331) + \
            0.00391838 * self.humidity**1.5 * math.atan(0.023101 * self.humidity) - 4.686035


class Location(BaseModel):
    id: str
    address: str
    name: str
    index: int
    latitude: float
    longitude: float
    distance: int
    time: int
    tz: str
    currentConditions: VC_Weather_data | None = VC_Weather_data()
    values: list[VC_Weather_data] | None = []


class Column(BaseModel):
    id: str
    name: str
    type: int
    unit: str | None = None


class VC_weather_model(BaseModel):
    columns: dict[str, Column] | None = None
    remainingCost: int | None = None
    queryCost: int | None = None
    location: Location | None = None
    message: str | None = None
    errorCode: int | None = None
    sessionId: str | None = None
