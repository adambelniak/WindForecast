from wind_forecast.datasets.BaseDataset import BaseDataset


class SequenceWithGFSDataset(BaseDataset):
    """Characterizes a dataset for PyTorch"""

    def __init__(self, synop_inputs, gfs_target_data, synop_targets, all_gfs_input_data=None):
        super().__init__()
        """Initialization"""
        self.all_gfs_input_data = all_gfs_input_data
        if all_gfs_input_data is not None:
            self.data = list(
                zip(zip(synop_inputs, all_gfs_input_data, gfs_target_data), synop_targets))
        else:
            self.data = list(zip(zip(synop_inputs, gfs_target_data), synop_targets))

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.data)

    def __getitem__(self, index):
        """Generates one sample of data"""

        sample, label = self.data[index][0], self.data[index][1]
        if self.all_gfs_input_data is not None:
            inputs, all_gfs_inputs, gfs_targets = sample[0], sample[1], sample[2]
            return inputs, all_gfs_inputs, gfs_targets, label

        x, gfs_targets = sample[0], sample[1]
        return x, gfs_targets, label
