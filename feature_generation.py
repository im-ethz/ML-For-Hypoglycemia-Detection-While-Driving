import os
import warnings
from typing import Iterable

import pandas as pd
import numpy as np
import glob

from joblib import Parallel, delayed
from tqdm import tqdm, trange
from datetime import timedelta

import sys
from custom_feature_preprocessing import CUSTOM_CAN, CUSTOM_PHASES, CUSTOM_INTERVENTION
from remove_intervention import distance, remove_intervention

from aggregation_functions import FUNCTIONS

COLUMN_RENAMES = {
    "VehicleSpeed": "velocity",
    "SteeringWheelAngle": "steer",
    "SteeringWheelAngularVelocity": "steer_vel",
    "BrakingPressure": "brake",
    "PedalForce": "gas",
    "LongitudinalAcceleration": "acc",
    "LateralAcceleration": "latacc",
}


def interpolate_timestamps(df):
    """
    Interpolate blood glucose values with same frequency as CAN data was recorded (50Hz).
    """
    interpolate_columns = ["bg_biosen", "bg_contour"]
    resampler = df.resample("0.02S")
    interpolated = (
        resampler.interpolate(method="time")[interpolate_columns]
        .add_suffix("_interpolate")
        .round(2)
    )
    return pd.concat([interpolated], axis=1)


def preprocess_bg_data(df):
    """
    Changes index to timestamp index,
    drops all columns except bg_biosen, bg_contour,
    drops bg values where valid == False.
    """

    subject = df["subject_id"].iloc[0]

    df.index = pd.to_datetime(df["timestamp"])
    df.index = df.index.tz_convert("Europe/Zurich")
    df = df[["bg_biosen", "bg_contour", "cgm", "valid"]]
    df = df.where(df["valid"].fillna(True)).drop("valid", axis=1).dropna(how="all")

    if df["cgm"].isna().all():
        print(f"No CGM values for subject {subject}, setting to -1")
        df["cgm"] = -1.0

    return df


def get_can_features(subject, data: pd.DataFrame, features, bg, epoch_width: int = 60):
    input_data = data.copy()

    inputs = trange(0, len(input_data) - 1, 50, desc=str(subject))

    results = Parallel(n_jobs=1)(
        delayed(__get_can_stats)(
            input_data, features=features, bg=bg, epoch_width=epoch_width, i=k
        )
        for k in inputs
    )
    results = pd.DataFrame(list(filter(None, results)))
    results.set_index("datetime", inplace=True)
    results.sort_index(inplace=True)

    return results


def get_stats(data, key_prefix: str = None):
    data_nans = data.isna().sum()
    if data_nans > 0:
        warnings.warn(f"input data contains {data_nans} NaNs which will be removed")
    data = data.dropna()
    data = np.asanyarray(data)

    results = {}
    try:
        if len(data) > 0:
            for key, value in FUNCTIONS.items():
                results[key] = value(data)
        else:
            for key in FUNCTIONS.keys():
                results[key] = np.nan
    except Exception as e:
        print(e)

    if key_prefix is not None:
        results = {key_prefix + "_" + k: v for k, v in results.items()}
    return results


def __get_can_stats(data: pd.DataFrame, features, bg, epoch_width: int, i: int):
    epoch_width = timedelta(seconds=epoch_width)
    requirement_1 = data.index[i + 1] - data.index[i] <= epoch_width
    requirement_2 = data.index[-1] - data.index[i] >= epoch_width
    if requirement_1 and requirement_2:
        min_timestamp = data.index[i]
        max_timestamp = min_timestamp + epoch_width

        results = {"datetime": max_timestamp}

        relevant_data = data.loc[
            (data.index >= min_timestamp) & (data.index < max_timestamp)
        ]

        for column in features:
            column_results = get_stats(relevant_data[column], f"can_{column}")
            results.update(column_results)

        # BG: get values at end of window
        if len(bg) > 0:
            results.update(relevant_data.iloc[-1][bg].to_dict())

        return results
    else:
        return None


def load_subject(
    subject: int,
    data_folder: str,
    window_size_sec: int,
    remove_intervention_phase: bool = True,
):
    folder = glob.glob(f"{data_folder}/202*_{subject}/")[0]

    study_part = 1 if subject < 300 else 2

    can_data = pd.read_parquet(folder + "output/canlogger/can-aggregated.parquet")
    can_data[list(COLUMN_RENAMES.keys())] = can_data[list(COLUMN_RENAMES.keys())].apply(
        lambda x: pd.to_numeric(x, errors="raise", downcast="float")
    )  # make sure we have floats for the needed signals

    phase_data = pd.read_csv(folder + "output/driving/phases.csv", parse_dates=[0, 1])
    if study_part == 1:  # study 1
        gps_data = pd.read_parquet(folder + "output/candump/gps.parquet")

        if subject in CUSTOM_PHASES.keys():
            phase_data = CUSTOM_PHASES[subject](phase_data)
            print("Preprocessed phase data for subject", subject)
        elif subject in CUSTOM_CAN.keys():
            can_data = CUSTOM_CAN[subject](folder, can_data, phase_data)
            print("Preprocessed CAN data for subject", subject)
        # else:
        #    print('No preprocessing necessary for subject', subject)

        if remove_intervention_phase:
            if subject in CUSTOM_INTERVENTION.keys():
                phase_data = CUSTOM_INTERVENTION[subject](
                    gps_data, phase_data, int(subject)
                )
            else:
                phase_data = remove_intervention(gps_data, phase_data, int(subject))

    can_data.rename(columns=COLUMN_RENAMES, inplace=True)
    data = can_data.loc[:, list(COLUMN_RENAMES.values())]
    data["gas_vel"] = data["gas"].diff().fillna(0)
    data["brake_vel"] = data["brake"].diff().fillna(0)
    data["steer_vel"] = data["steer_vel"].astype(float)
    data["steer_vel_abs"] = data["steer_vel"].abs()
    data["steer_abs"] = data["steer"].abs()
    data["acc_abs"] = data["acc"].abs()

    data[data.columns.difference(["phase", "scenario"])] = data[
        data.columns.difference(["phase", "scenario"])
    ].astype("float32")

    # inteprolate and add bg_data
    if os.path.exists(folder + "bg/bg.csv"):
        bg_data = pd.read_csv(folder + "bg/bg.csv")
        bg_data = preprocess_bg_data(bg_data)
        interpolation = "ffill"  # 'ffill' or 'time'
        assert interpolation in ["ffill", "time"]
        bg_data = (
            bg_data.asfreq("0.02S")
            .interpolate(method=interpolation)
            .interpolate(method="bfill")
            .astype("float32")
        )  # bfill for rows in beginning
        data = pd.merge_asof(
            data,
            bg_data,
            direction="nearest",
            tolerance=timedelta(seconds=window_size_sec),
            left_index=True,
            right_index=True,
        )
    else:
        raise ("No BG data for subject ", subject)
        # bg_data = pd.DataFrame()

    data["phase"] = 0
    data["scenario"] = None

    end = "end" if study_part == 1 else "last_start_pass"
    for idx, phase in phase_data.iterrows():
        data.loc[
            ((data.index > phase["start"]) & (data.index <= phase[end])),
            ["phase", "scenario"],
        ] = [phase["phase"], phase["scenario"]]

    data.dropna(how="any", inplace=True)

    features = []
    for phase in [1, 2, 3, 4]:
        relevant_data = data[data["phase"] == phase]

        for scenario in relevant_data["scenario"].unique():
            input_data = relevant_data[relevant_data["scenario"] == scenario].drop(
                columns=["phase", "scenario"]
            )
            if len(input_data) == 0:
                print(
                    "No data available for subject",
                    subject,
                    "in phase",
                    phase,
                    "and scenario",
                    scenario,
                )
                continue
            df = get_can_features(
                subject,
                input_data,
                features=data.columns.difference(
                    ["phase", "scenario"] + bg_data.columns.to_list()
                ),
                bg=bg_data.columns,
                epoch_width=window_size_sec,
            )
            df["phase"] = phase
            df["scenario"] = scenario
            df["subject_id"] = subject

            features.append(df)

    features = pd.concat(features)
    features["scenario"] = features["scenario"].str.title()
    features.index = features.index.floor(freq="s")
    return features


def generate_features(
    data_folder: str,
    subjects: list,
    window_size_sec: int,
    remove_intervention_phase: bool = True,
):
    """
    Generates features with aggregation functions from aggregation_functions.py using
    a sliding window approach with length window_size and a shift of 1s.
    """

    print(f"Generating features for {len(subjects)} subjects: {sorted(subjects)}")
    with Parallel(n_jobs=min(32, len(subjects))) as parallel:
        features = parallel(
            delayed(load_subject)(
                subject, data_folder, window_size_sec, remove_intervention_phase
            )
            for subject in subjects
        )

    features = pd.concat(features)
    return features
