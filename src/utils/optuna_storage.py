import os
import optuna
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Single storage file for all jobs
STORAGE_PATH = "optuna_storage/optimization.db"
STORAGE_URL = f"sqlite:///{STORAGE_PATH}"

def get_study(job_name: str, sub_job_name: str, direction: str = 'minimize', use_existing: bool = False) -> optuna.Study:
    """
    Get or create an Optuna study for a specific job and sub-job.
    
    Args:
        job_name: Name of the main job (e.g., 'llm_full_features')
        sub_job_name: Name of the sub-job (e.g., 'lstm_full')
        direction: Optimization direction ('minimize' or 'maximize')
        use_existing: Whether to use existing storage
    
    Returns:
        optuna.Study: The study object for the specified job and sub-job
    """
    # Create storage directory if it doesn't exist
    os.makedirs('optuna_storage', exist_ok=True)
    
    # Create a unique study name combining job and sub-job
    study_name = f"{job_name}-{sub_job_name}"
    
    # Create or load the study
    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        storage=STORAGE_URL,
        load_if_exists=use_existing
    )
    
    logger.info(f"Using study '{study_name}' in storage {STORAGE_URL}")
    return study

def get_best_params(job_name: str, sub_job_name: str) -> dict:
    """
    Get the best parameters for a specific job and sub-job.
    
    Args:
        job_name: Name of the main job
        sub_job_name: Name of the sub-job
    
    Returns:
        dict: Best parameters for the specified job and sub-job
    """
    study_name = f"{job_name}_{sub_job_name}"
    study = optuna.load_study(study_name=study_name, storage=STORAGE_URL)
    return study.best_params

def get_all_studies() -> list:
    """
    Get a list of all studies in the storage.
    
    Returns:
        list: List of study names
    """
    return optuna.get_all_study_names(storage=STORAGE_URL) 