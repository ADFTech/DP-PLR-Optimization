from sqlalchemy.orm import declarative_base, relationship, backref
from sqlalchemy import Column, Float, Integer, String, DateTime, ForeignKey, Index

Base = declarative_base()

class Weather_data(Base):
    """
    Represents a row in the weather_data table

    Relationships:
        Has one location via self.location_id
    """
    __abstract__ = True

    id = Column(Integer, primary_key=True)
    wdir = Column(Float, nullable = True)
    uvindex = Column(Float, nullable = True)
    preciptype = Column(String, nullable = True)
    cin = Column(Float, nullable = True)
    cloudcover = Column(Float, nullable = True)
    pop = Column(Float, nullable = True)
    mint = Column(Float, nullable = True)
    datetime = Column(DateTime, nullable = True, unique = True)
    precip = Column(Float, nullable = True)
    solarradiation = Column(Float, nullable = True)
    dew = Column(Float, nullable = True)
    humidity = Column(Float, nullable = True)
    temp = Column(Float, nullable = True)
    maxt = Column(Float, nullable = True)
    visibility = Column(Float, nullable = True)
    wspd = Column(Float, nullable = True)
    severerisk = Column(Float, nullable = True)
    solarenergy = Column(Float, nullable = True)
    heatindex = Column(Float, nullable = True)
    snowdepth = Column(Float, nullable = True)
    sealevelpressure = Column(Float, nullable = True)
    snow = Column(Float, nullable = True)
    wgust = Column(Float, nullable = True)
    conditions = Column(String, nullable = True)
    windchill = Column(Float, nullable = True)
    cape = Column(Float, nullable = True)
    sunrise = Column(String, nullable = True)
    icon = Column(String, nullable = True)
    moonphase = Column(Float, nullable = True)
    sunset = Column(String, nullable = True)
    stull_wetbulb_approx = Column(Float, nullable = True)
    location_id = Column(String, ForeignKey("location.id"), nullable=False)


class Historical_weather(Weather_data):
    """
    Represents a row in the historical_weather table

    Relationships:
        Has many weather_data via Weather_data.location_id

    Indexes:
        idx_historical_weather_datetime
        idx_historical_weather_location
    """
    __tablename__ = "historical_weather"
    __table_args__ = (
        Index('idx_historical_weather_datetime', 'datetime'),
        Index('idx_historical_weather_location', 'location_id'),
    )
    location = relationship(
        "Location",
        backref = backref("historical_weather", lazy='dynamic')
    )


class Location(Base):
    """
    Represents a row in the location table

    Relationships:
        Has many weather_data via Weather_data.location_id
    """
    __tablename__ = "location"

    id = Column(String, primary_key=True)
    address = Column(String, nullable = False)
    name = Column(String, nullable = False)
    index = Column(Integer, nullable = False)
    latitude = Column(Float, nullable = False)
    longitude = Column(Float, nullable = False)
    distance = Column(Float, nullable = False)
    time = Column(Integer, nullable = False)
    tz = Column(String, nullable = False)
