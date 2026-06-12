from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from granitewxc.utils.config import ExperimentConfig
from granitewxc.utils.distributed import is_main_process
from granitewxc.datasets.eccc  import EcccHrdpsGdpsDataset


def get_dataloaders_eccc(config: ExperimentConfig, test: bool = False) -> tuple[DataLoader, ...]:
    """
    Args:
        config: Experiment configuration. Contains configuration parameters for model.
        test: If True, only the test DataLoader is returned. If False, train, valid, and test DataLoaders are returned.
    Returns:
        Tuple of data loaders: (training loader, validation loader).
    """

    ds_kwargs = dict(
        json_static_var_path = config.data.static_data_index,
        surface_vars = config.data.input_surface_vars, 
        vertical_pres_vars = config.data.vertical_pres_vars,
        vertical_level1_vars = config.data.vertical_level1_vars,
        vertical_level2_vars = config.data.vertical_level2_vars,
        other_vars = config.data.other,
        static_vars = config.data.input_static_surface_vars,
        output_vars = config.data.output_vars,
        downsample_factor = config.data.downsample_factor,
        n_random_windows = config.data.n_random_windows,
        crop_factor = config.data.crop_factor,
    )

    dl_kwargs = dict(
        batch_size=config.batch_size,
        num_workers=config.dl_num_workers,
        prefetch_factor=config.dl_prefetch_size,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )

    if test:
        test_dataset = EcccHrdpsGdpsDataset(json_file_path = config.data.data_test_index, test=True, **ds_kwargs)
        test_loader = DataLoader(dataset=test_dataset, **dl_kwargs)

        if is_main_process():
            print(f"--> Test samples: {len(test_dataset):,.0f}")

        return test_loader

    train_dataset = EcccHrdpsGdpsDataset(json_file_path = config.data.data_training_index, **ds_kwargs)
    valid_dataset = EcccHrdpsGdpsDataset(json_file_path = config.data.data_val_index, **ds_kwargs)
    
    try:
        train_loader = DataLoader(
            dataset=train_dataset,
            sampler=DistributedSampler(train_dataset, shuffle=True, drop_last=True),
            **dl_kwargs,
        )
        valid_loader = DataLoader(
            dataset=valid_dataset,
            sampler=DistributedSampler(valid_dataset, drop_last=False),
            **dl_kwargs,
        )
    except:
        train_loader = DataLoader(
            dataset=train_dataset,
            **dl_kwargs,
        )
        valid_loader = DataLoader(
            dataset=valid_dataset,
            **dl_kwargs,
        )

    if is_main_process():
        print(f"--> Training batches: {len(train_loader):,.0f}")
        print(f"--> Validation batches: {len(valid_loader):,.0f}")
        print(f"--> Training samples: {len(train_dataset):,.0f}")
        print(f"--> Validation samples: {len(valid_dataset):,.0f}")

    return (
        train_loader,
        valid_loader,
    )