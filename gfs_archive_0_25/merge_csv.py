import os
import pandas as pd
import csv
import argparse
from os.path import isfile, join

csv_dirs_to_parameter_map = [
    {
        'parameter': '10m U-wind, m/s',
        'dir': '../../../grib_csv/uwind'
    },
    {
        'parameter': '10m V-wind, m/s',
        'dir': '../../../grib_csv/vwind'
    },
    {
         'parameter': 'Gusts, m/s',
         'dir': '../../../grib_csv/gusts'
    },
    {
       'parameter': '2m temperature, K',
       'dir': '../../../grib_csv/2mtemp'
    }
   # {
    #    'parameter': 'Convective precipitation rate, mm/h',
     #   'dir': '../../../grib_csv_cprat'
    #},
]
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dirs', help='Input directories with csv files. Names of files should be in format YYYY-mm-dd-HHZ.csv',
                        nargs='+', required=True)
    parser.add_argument('--names', help='Names of the parameters for each input csv', nargs='+', required=True)
    parser.add_argument('--out', help='Output directory for csv files', default='')

    args = parser.parse_args()

    if len(args.dirs) != len(args.names):
        parser.error("Length of --dirs and --names arguments must match!")

    input_dirs = args.dirs
    files_in_dir = []
    for dir in input_dirs:
        files_in_dir.append([f for f in os.listdir(dir) if isfile(join(dir, f))])

    common_files = files_in_dir[0]
    for dir in files_in_dir:
        common_files = list(set(common_files).intersection(set(dir)))

    for f in common_files:
        print("Writing file " + f)
        data = {'date': pd.read_csv(open(join(input_dirs[0], f)),
                                    header=0,
                                    names=['date', 'a'])['date'].values.tolist()}

        input_dirs_and_params = list(
            map(lambda dir_name: {'dir': dir_name[0], 'param': dir_name[1]}, list(zip(args.dirs, args.names))))

        for dir_param in input_dirs_and_params:
            print('Reading from directory ' + dir_param['dir'])
            assert os.path.exists(join(dir_param['dir'], f))
            in_file = open(join(dir_param['dir'], f))
            df = pd.read_csv(in_file, header=0, names=['date', dir_param['param']])
            data[dir_param['param']] = df[dir_param['param']].values.tolist()
            in_file.close()
        out_dir = args.out
        out_csv = join(out_dir, f)
        if os.path.exists(out_dir) is not True:
            os.makedirs(out_dir)
        with open(out_csv, 'w') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(data.keys())
            writer.writerows(zip(*data.values()))
            outfile.close()
