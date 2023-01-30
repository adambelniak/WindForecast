from typing import Optional

from wind_forecast.config.register import Config
from wind_forecast.datamodules.Sequence2SequenceDataModule import Sequence2SequenceDataModule
from wind_forecast.datasets.Sequence2SequenceWithGFSDataset import Sequence2SequenceWithGFSDataset


class Sequence2SequenceDataModuleForGFSComparison(Sequence2SequenceDataModule):

    def __init__(
            self,
            config: Config
    ):
        super().__init__(config)

    def setup(self, stage: Optional[str] = None):
        if self.initialized:
            self.log_dataset_info()
            return
        if self.get_from_cache(stage):
            self.log_dataset_info()
            return

        self.prepare_dataset_for_gfs()
        dataset = Sequence2SequenceWithGFSDataset(self.config, self.synop_data, self.gfs_data, self.data_indices,
                                                  self.synop_feature_names, self.gfs_features_names)

        if len(dataset) == 0:
            raise RuntimeError("There are no valid samples in the dataset! Please check your run configuration")

        dataset.set_mean(self.synop_mean)
        dataset.set_std(self.synop_std)
        self.split_dataset(self.config, dataset, self.sequence_length)
        self.log_dataset_info()
