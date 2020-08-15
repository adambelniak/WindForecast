import pygrib
from utils import get_nearest_coords
from scipy import interpolate


def fetch_data_from_grib(grib_file, gfs_variables, latitude=0., longitude=0.):
    print("Fetching gfs data from file " + grib_file)
    gr = pygrib.open(grib_file)
    data = {}
    for variable in gfs_variables:
        try:
            if 'typeOfLevel' in variable:
                message = gr.select(shortName=variable['shortName'], typeOfLevel=variable['typeOfLevel'],
                                    level=variable['level'])[0]
            else:
                message = gr.select(shortName=variable['shortName'])[0]
            coords = get_nearest_coords(latitude, longitude)
            values = message.data(lat1=coords[0][0],
                                  lat2=coords[0][1],
                                  lon1=coords[1][0],
                                  lon2=coords[1][1])
            # pygrib needs to have latitude from smaller to larger, but returns the values north-south,
            # so from larger lat to smaller lat 🙁 Thus, reversing 'data' array
            interpolated_function_for_values = interpolate.interp2d(coords[0], coords[1], values[0][::-1])
            data[variable['fullName']] = interpolated_function_for_values(latitude, longitude).item()
        except:
            data[variable['fullName']] = 0.0
    return data