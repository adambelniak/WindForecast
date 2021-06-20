import pygrib
from gfs_archive_0_25.utils import get_nearest_coords
from scipy import interpolate


def fetch_data_from_grib(grib_file, gfs_parameters, latitude=0., longitude=0.):
    print("Fetching gfs data from file " + grib_file)
    gr = pygrib.open(grib_file)
    data = {}
    for parameter in gfs_parameters:
        try:
            if 'typeOfLevel' in parameter:
                message = gr.select(shortName=parameter['shortName'], typeOfLevel=parameter['typeOfLevel'],
                                    level=parameter['level'])[0]
            else:
                message = gr.select(shortName=parameter['shortName'])[0]
            coords = get_nearest_coords(latitude, longitude)
            values = message.data(lat1=coords[0][0],
                                  lat2=coords[0][1],
                                  lon1=coords[1][0],
                                  lon2=coords[1][1])
            # pygrib needs to have latitude from smaller to larger, but returns the values north-south,
            # so from larger lat to smaller lat 🙁 Thus, reversing 'data' array
            interpolated_function_for_values = interpolate.interp2d(coords[0], coords[1], values[0][::-1])
            data[parameter['fullName']] = interpolated_function_for_values(latitude, longitude).item()
        except:
            print("Data not found in grib file. Setting it to 0.0.")
            data[parameter['fullName']] = 0.0
    gr.close()
    return data
