import optuna

storage = "sqlite:///optuna_storage/optimization.db"  # Update this path!

study_names = [
    "default_job-default_sub_job"
]

for name in study_names:
    try:
        optuna.delete_study(study_name=name, storage=storage)
        print(f"Deleted study: {name}")
    except KeyError:
        print(f"Study not found: {name}")