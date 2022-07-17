STATION_CODE = (0, "station_code")
STATION_NAME = (1, "station_name")
YEAR = (2, 'year')
MONTH = (3, 'month')
DAY = (4, 'day')
HOUR = (5, 'hour')
VISIBILITY = (17, "visibility")
CLOUD_COVER = (21, "cloud_cover")
DIRECTION_COLUMN = (23, 'wind_direction')
VELOCITY_COLUMN = (25, 'wind_velocity')
GUST_COLUMN = (27, 'wind_gust')
TEMPERATURE = (29, 'temperature')
HUMIDITY = (37, "humidity")
DEW_POINT = (39, "dew_point")
PRESSURE = (41, 'pressure')
PRESSURE_AT_SEA_LEVEL = (43, 'pressure_at_sea_level')
PRECIPITATION_6H = (48, 'precipitation_6h')
PRECIPITATION_TYPE = (50, 'precipitation_type')
CURRENT_WEATHER = (52, 'current_weather')
LOWER_CLOUDS = (54, 'lower_clouds')
# Auto station features
AUTO_TEMPERATURE = ('B00300S', 'temperature')
AUTO_TEMPERATURE_SURFACE = ('B00305A', 'temperature_surface')
AUTO_WIND_DIRECTION = ('B00202A', 'wind_direction')
AUTO_WIND = ('B00702A', 'wind_velocity')
AUTO_GUST = ('B00703A', 'wind_gust')
AUTO_10_MIN_PRECIPITATION = ('B00608S', 'precipitation')
AUTO_HUMIDITY = ('B00802A', 'humidity')

SYNOP_FEATURES = [
    YEAR,
    MONTH,
    DAY,
    HOUR,
    VISIBILITY,
    CLOUD_COVER,
    DIRECTION_COLUMN,
    VELOCITY_COLUMN,
    GUST_COLUMN,
    TEMPERATURE,
    HUMIDITY,
    DEW_POINT,
    PRESSURE,
    PRESSURE_AT_SEA_LEVEL,
    PRECIPITATION_6H,
    PRECIPITATION_TYPE,
    CURRENT_WEATHER,
    LOWER_CLOUDS
]

SYNOP_TRAIN_FEATURES = [
    VISIBILITY,
    CLOUD_COVER,
    DIRECTION_COLUMN,
    VELOCITY_COLUMN,
    GUST_COLUMN,
    TEMPERATURE,
    HUMIDITY,
    DEW_POINT,
    PRESSURE,
    PRECIPITATION_6H,
    LOWER_CLOUDS
]

AUTO_STATION_FEATURES = [
    AUTO_TEMPERATURE,
    AUTO_TEMPERATURE_SURFACE,
    AUTO_WIND_DIRECTION,
    AUTO_WIND,
    AUTO_GUST,
    AUTO_10_MIN_PRECIPITATION,
    AUTO_HUMIDITY
]

# Synop features which will be split into sin and cos during normalization phase
SYNOP_PERIODIC_FEATURES = [
    {
        'column': DIRECTION_COLUMN,
        'min': 0,
        'max': 360
    }
]