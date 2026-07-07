"""Centralized paths and constants configuration.

Example:
    import src.path_and_constants
    paths = src.path_and_constants.Paths()
"""

class Paths:
    """Project directory and file paths."""

    train_path: str = "data/crosstalk_train.parquet"
    test_path: str = "data/crosstalk_test_inputs.parquet"
    model_path: str = "models/best_model.pkl"
    submission_path: str = "submission.csv"


class Constants:
    """Project-wide metadata constants."""

    file_ids: dict[str, str] = {
        "data/crosstalk_train.parquet": "11S5p0QgP1X9rOFiIjNSLydLenJwm7hle",
        "data/crosstalk_test_inputs.parquet": "15iMvnmIraM-geCI-vG9iR5naliWfh5tA",
    }
