import os
import warnings

warnings.filterwarnings("ignore")

try:  # Speedup for intel processors
    from sklearnex import patch_sklearn

    patch_sklearn()
except:
    pass

import datetime
import numpy as np
import pandas as pd
from sklearn import preprocessing

from feature_generation import generate_features
from feature_selection import calculate_cv_score

np.random.seed(0)

############# USER SETTING
WINDOW_SIZE = 60

STUDY_1_SUBJECTS = [
    201,
    202,
    204,
    205,
    206,
    207,
    208,
    211,
    212,
    213,
    214,
    215,
    216,
    217,
    219,
    220,
    221,
    222,
    223,
    224,
    225,
]
STUDY_2_SUBJECTS = [401, 402, 403, 404, 405, 406, 407, 409, 410, 411, 412]

# REMOVE
STUDY_1_SUBJECTS.remove(220)  # missing ET data in phase 3
STUDY_2_SUBJECTS.remove(407)  # missing ET data in phase 1

LOSO_SUBJECTS = STUDY_1_SUBJECTS + STUDY_2_SUBJECTS
#############


def load_data(window_size_sec, reload_can=False, reload_et=False):
    data_filename = f"./data/all_data_{window_size_sec:03d}.parquet"
    if reload_can or reload_et or not os.path.exists(data_filename):
        print("Reloading data...")
        if os.uname()[1] == "mtec-im-gpu01":
            DATA_FOLDER = "/headwind/field-study/"
        else:
            raise "Need to specify data folder for hosts other than mtec-im-gpu01"

        can_filename = f"./data/all_can_{window_size_sec:03d}.parquet"
        if reload_can or not os.path.exists(can_filename):
            print("Preprocessing CAN data")

            X = generate_features(
                DATA_FOLDER, STUDY_1_SUBJECTS + STUDY_2_SUBJECTS, window_size_sec
            )
            X.sort_index(inplace=True)
            # filter non-finite columns, some entropy cols might be nan
            X.replace([np.inf, -np.inf], np.nan, inplace=True)
            X.dropna(axis="columns", how="any", inplace=True)
            X.to_parquet(
                can_filename, allow_truncated_timestamps=True, coerce_timestamps="ms"
            )
        else:
            X = pd.read_parquet(can_filename)

        et_filename = f"./data/all_et_{window_size_sec:03d}.parquet"
        if reload_et or not os.path.exists(et_filename):
            print("Loading ET file")

            et_data = pd.read_parquet(f"./data/all_probands_{window_size_sec}.parquet")

            # ATTENTION: ET file uses beginning of time window as index, but CAN uses end. Unify to end.
            et_data.index += datetime.timedelta(seconds=window_size_sec)

            et_data.sort_index(inplace=True)
            et_data["phase"] = et_data["phase"].astype(int)
            et_data = et_data[et_data["phase"].isin([1, 2, 3, 4])]
            et_data.insert(0, "subject_id", et_data["id"].astype(int))
            et_data.drop(columns=["id"], inplace=True)
            et_data["scenario"] = et_data["scenario"].str.title()
            et_data.index = et_data.index.floor(freq="s")
            et_data = et_data[~et_data.index.duplicated(keep="first")]

            et_data.to_parquet(
                et_filename, allow_truncated_timestamps=True, coerce_timestamps="ms"
            )
        else:
            et_data = pd.read_parquet(et_filename)

     
        # filter all entropy region features
        et_data = et_data.filter(regex="^(?!entropyregion)")
        X = pd.merge_asof(
            X,
            et_data.iloc[:, 6:].add_prefix("et_"),
            left_index=True,
            right_index=True,
            direction="nearest",
            tolerance=datetime.timedelta(seconds=1),
        )
        X.dropna(how="any", inplace=True)
        X.to_parquet(
            data_filename, allow_truncated_timestamps=True, coerce_timestamps="ms"
        )
    else:
        print(f"Loading overall data from file {data_filename}")
        X = pd.read_parquet(data_filename)

    X.columns = X.columns.str.replace("azimuth_elevation", "overall")
    return X


def run_evaluation(window_size_sec):
    X = load_data(window_size_sec, reload_can=False, reload_et=False)
    X[["train", "test"]] = True  # no holdout for now, can train/test on all data
    X = X[X["phase"].isin([1, 3])]  # only use phases 1 and 3

    # generate label based on bg_sensor readings
    bg_sensor = "bg_biosen"
    if bg_sensor in X.columns and not X[bg_sensor].isna().any():
        X["y_30"], X["y_39"] = X[bg_sensor] < 3.0, X[bg_sensor] < 3.9
        label_column = "y_39"
    else:
        label_column = "label"
        X[label_column] = X["phase"] == 3

    # ensure we only use eu in phase 1 and hypo in phase 3
    X = X[
        ((X["phase"] == 1) & (X[label_column] == False))
        | ((X["phase"] == 3) & (X[label_column] == True))
    ]

    evaluate_indices_studies = [
        X["test"] & X["subject_id"].isin(LOSO_SUBJECTS),
        X["test"] & X["subject_id"].isin(STUDY_1_SUBJECTS),
        X["test"] & X["subject_id"].isin(STUDY_2_SUBJECTS),
    ]

    train_test_configs = [
        (
            X["train"] & X["subject_id"].isin(LOSO_SUBJECTS),
            X["test"] & X["subject_id"].isin(LOSO_SUBJECTS),
            evaluate_indices_studies,
        ),  # MIXED model
        # (X['train'] & X['subject_id'].isin(STUDY_1_SUBJECTS), X['test'] & X['subject_id'].isin(STUDY_1_SUBJECTS), evaluate_indices_studies),  # MODERATE model
        # (X['train'] & X['subject_id'].isin(STUDY_2_SUBJECTS), X['test'] & X['subject_id'].isin(STUDY_2_SUBJECTS), evaluate_indices_studies),  # MILD model
    ]

    X_save = X.copy()
    for train_indices, test_indices, evaluate_indices in train_test_configs:
        all_evaluate_indices = pd.concat(evaluate_indices, axis=1).any(axis=1)
        all_relevant_indices = train_indices | test_indices | all_evaluate_indices
        X = X_save[all_relevant_indices]
        train_indices = train_indices[all_relevant_indices]
        test_indices = test_indices[all_relevant_indices]

    
        # Loop over modalities
        for modality in ["CAN+ET", "CAN", "ET"]:
            features = []

            if "CAN" in modality:
                features.extend(
                    [
                        "can_steer_vel_n_sign_changes",
                        "can_steer_n_sign_changes",
                        "can_acc_n_sign_changes",
                        "can_gas_iqr",
                        "can_brake_iqr",
                        "can_velocity_iqr",
                    ]
                )
            if "ET" in modality:
                features.extend(
                    [
                        "et_iqrange_acceleration_yaw",
                        "et_iqrange_acceleration_roll",
                        "et_iqrange_acceleration_pitch",
                        "et_median_speed_head",
                        "et_median_acceleration_head",
                        "et_median_angle_change",
                        "et_median_acceleration",
                        "et_iqrange_angle_change",
                        "et_iqrange_acceleration",
                    ]
                )


            calculate_cv_score(
                X=X[features],
                y=X[label_column],
                groups=X["subject_id"],
                scenarios=X["scenario"],
                train_indices=train_indices,
                test_indices=test_indices,
                features=features,
                desc=modality.lower(),
                bg=X[bg_sensor],
                store_files=True,
            )

    return


def generate_can_files():
    DATA_FOLDER = "/headwind/field-study/"
    
    for window_size_sec in [5, 10, 15, 30, 45, 60, 75, 90]:
        can_filename = f"./data/all_can_{window_size_sec:03d}.parquet"

        print(f"Processing window size {window_size_sec}: {can_filename}")
        if not os.path.exists(can_filename):
            X = generate_features(
                DATA_FOLDER, STUDY_1_SUBJECTS + STUDY_2_SUBJECTS, window_size_sec
            )
            X.sort_index(inplace=True)
            X.replace([np.inf, -np.inf], np.nan, inplace=True)
            X.dropna(axis="columns", how="any", inplace=True)
            X.to_parquet(
                can_filename, allow_truncated_timestamps=True, coerce_timestamps="ms"
            )
        else:
            print(f"File {can_filename} already exists.")


if __name__ == "__main__":
    run_evaluation(window_size_sec=WINDOW_SIZE)
