from datetime import datetime
from pydantic import BaseModel

class Data_field(BaseModel):
    Value: float | None = None
    Unit: str | None = None
    UnitType: int | None = None
    Phrase: str | None = None
    Degrees: int | None = None
    Localized: str | None = None
    English: str | None = None


class Wind_data_field(BaseModel):
    Speed: Data_field
    Direction: Data_field | None = None


class Accu_weather_data(BaseModel):
    DateTime: datetime
    EpochDateTime: int
    WeatherIcon: int
    IconPhrase: str
    HasPrecipitation: bool
    IsDaylight: bool
    Temperature: Data_field
    RealFeelTemperature: Data_field
    RealFeelTemperatureShade: Data_field
    WetBulbTemperature: Data_field
    WetBulbGlobeTemperature: Data_field
    DewPoint: Data_field
    Wind: Wind_data_field
    WindGust: Wind_data_field
    RelativeHumidity: int
    IndoorRelativeHumidity: int
    Visibility: Data_field
    Ceiling: Data_field
    UVIndex: int
    UVIndexText: str
    PrecipitationProbability: int
    ThunderstormProbability: int
    RainProbability: int
    SnowProbability: int
    IceProbability: int
    TotalLiquid: Data_field
    Rain: Data_field
    Snow: Data_field
    Ice: Data_field
    CloudCover: int
    Evapotranspiration: Data_field
    SolarIrradiance: Data_field
    MobileLink: str
    Link: str


class Accu_weather_model(BaseModel):
    weather_data: list[Accu_weather_data]
