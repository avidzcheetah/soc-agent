# Responsibility: Create experiment folders and metadata

def get_next_experiment_dir(base_dir="experiments"):
    """
    Look up experiments base directory to assign next experiment_00x code.
    """
    pass

def create_experiment_layout(output_dir=None):
    """
    Construct subdirectories inside the allocated experiment directory path.
    """
    pass

def archive_experiment_metadata(config, dirs, dataset_sizes, best_epoch, best_macro_f1, best_val_loss, test_metrics, gpu_name, cuda_version):
    """
    Produce experiment.json metadata document and append details to central summary.csv file.
    """
    pass
