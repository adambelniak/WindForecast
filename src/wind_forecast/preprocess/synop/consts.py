YEAR = (2, 'year')
MONTH = (3, 'month')
DAY = (4, 'day')
HOUR = (5, 'hour')
DIRECTION_COLUMN = (23, 'wind_direction')
VELOCITY_COLUMN = (25, 'wind_velocity')
GUST_COLUMN = (27, 'wind_gust')
TEMPERATURE = (29, 'temperature')
PRESSURE = (41, 'pressure')
CURRENT_WEATHER = (52, 'current_weather')

lstm_features = [
    DIRECTION_COLUMN,
    VELOCITY_COLUMN,
    GUST_COLUMN,
    TEMPERATURE,
    PRESSURE,
    CURRENT_WEATHER
]