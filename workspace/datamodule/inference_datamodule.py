import lightning as L
from torch.utils.data import DataLoader

from .inference_dataset import InferenceCaseDataset, VideoInferenceCaseDataset, inference_case_collate


class InferenceDataModule(L.LightningDataModule):
    def __init__(
        self,
        is_video,
        image_folder,
        file_ending,
        num_channels,
        plans_manager,
        configuration_manager,
        dataset_json,
        work_dir=None,
    ):
        super().__init__()

        self.is_video = is_video
        self.image_folder = image_folder
        self.file_ending = file_ending
        self.num_channels = num_channels
        self.work_dir = work_dir

        self.pm = plans_manager
        self.cm = configuration_manager
        self.dataset_json = dataset_json

        self.dataset = None

    def setup(self, stage=None):
        if self.is_video:
            self.dataset = VideoInferenceCaseDataset(
                image_folder=self.image_folder,
                work_dir=self.work_dir,
                plans_manager=self.pm,
                configuration_manager=self.cm,
                dataset_json=self.dataset_json,
            )
        else:
            self.dataset = InferenceCaseDataset(
                image_folder=self.image_folder,
                file_ending=self.file_ending,
                num_channels=self.num_channels,
                plans_manager=self.pm,
                configuration_manager=self.cm,
                dataset_json=self.dataset_json,
            )

    def predict_dataloader(self):
        return DataLoader(
            self.dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=inference_case_collate
        )
