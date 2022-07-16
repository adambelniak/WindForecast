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
            return
        if self.get_from_cache(stage):
            return

        synop_inputs, all_gfs_input_data, gfs_target_data, all_gfs_target_data = self.prepare_dataset_for_gfs()
        dataset = Sequence2SequenceWithGFSDataset(self.config, self.synop_data, self.synop_data_indices, self.synop_feature_names,
                                                  gfs_target_data)

        if len(dataset) == 0:
            raise RuntimeError("There are no valid samples in the dataset! Please check your run configuration")

        dataset.set_mean(self.synop_mean)
        dataset.set_std(self.synop_std)
        self.split_dataset(self.config, dataset, self.sequence_length)