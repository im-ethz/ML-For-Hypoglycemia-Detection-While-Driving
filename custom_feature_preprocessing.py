import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob

from remove_intervention import distance, remove_intervention


def can_202(
    folder: str, can_log_data: pd.DataFrame, phase_data: pd.DataFrame
) -> pd.DataFrame:
    """
    CAN log failed to record the first 2min of phase 3, scenario highway;
    retrieve data from CAN dump.
    """

    can_log_data_filled = can_log_data.copy()

    can_dump_data = pd.read_parquet(f"{folder}output/candump/driving.parquet")

    phase_data["start"] = pd.to_datetime(phase_data["start"]).apply(
        lambda row: row.tz_convert("Europe/Zurich")
    )
    phase_data["end"] = pd.to_datetime(phase_data["end"]).apply(
        lambda row: row.tz_convert("Europe/Zurich")
    )

    phase_start = phase_data[
        (phase_data["phase"] == 3) & (phase_data["scenario"] == "highway")
    ].start.iloc[0]
    phase_end = phase_data[
        (phase_data["phase"] == 3) & (phase_data["scenario"] == "highway")
    ].end.iloc[0]

    relevant_data = can_dump_data.loc[phase_start:phase_end]
    relevant_data = relevant_data.resample("0.02S").ffill().dropna()

    # PedalForce in candump is in % (0-100) but we need it between 0-1 here to be aligned with the canlogger
    relevant_data["PedalForce"] = relevant_data["PedalForce"] / 100

    can_log_data_filled.loc[phase_start:phase_end, :] = relevant_data

    return can_log_data_filled


def can_211(
    folder: str, can_log_data: pd.DataFrame, phase_data: pd.DataFrame
) -> pd.DataFrame:
    """
    CAN log failed to record the VehicleSpeed for 18s in phase 2, scenario highway and rural;
    exact time: (09:54:09–09:54:27) and (10:25:47–10:26:05);
    retrieve data from CAN dump.
    """
    can_log_data_filled = can_log_data.copy()
    can_dump_data = pd.read_parquet(f"{folder}/output/candump/driving.parquet")

    phase_start = phase_data[
        (phase_data["phase"] == 1) & (phase_data["scenario"] == "highway")
    ].start.iloc[0]
    phase_end = phase_data[
        (phase_data["phase"] == 2) & (phase_data["scenario"] == "rural")
    ].end.iloc[0]

    relevant_data = can_dump_data.loc[phase_start:phase_end, "VehicleSpeed"]
    relevant_data = relevant_data.resample("0.02S").ffill().dropna()

    can_log_data_filled.loc[phase_start:phase_end, "VehicleSpeed"] = relevant_data

    return can_log_data_filled


def can_214(
    folder: str, can_log_data: pd.DataFrame, phase_data: pd.DataFrame
) -> pd.DataFrame:
    """
    CAN log failed to record phase 1, scenario city;
    retrieve data from CAN dump.
    """
    can_log_data_filled = can_log_data.copy()
    can_dump_data = pd.read_parquet(f"{folder}output/candump/driving.parquet")

    phase_start = phase_data[
        (phase_data["phase"] == 1) & (phase_data["scenario"] == "city")
    ].start.iloc[0]
    phase_end = phase_data[
        (phase_data["phase"] == 1) & (phase_data["scenario"] == "city")
    ].end.iloc[0]

    relevant_data = can_dump_data.loc[phase_start:phase_end]
    relevant_data = relevant_data.resample("0.02S").ffill().dropna()

    # PedalForce in candump is in % (0-100) but we need it between 0-1 here to be aligned with the canlogger
    relevant_data["PedalForce"] = relevant_data["PedalForce"] / 100

    can_log_data_filled.loc[phase_start:phase_end, :] = relevant_data

    return can_log_data_filled


def phase_204(phase_data: pd.DataFrame) -> pd.DataFrame:
    """
    Neither CAN dump nor CAN logger did work during the first 90s of phase 1, scenario rural;
    hence we cut this part out.
    """
    start_loc = phase_data.columns.get_loc("start")
    index_loc = phase_data[
        (phase_data["phase"] == 1) & (phase_data["scenario"] == "rural")
    ].index.item()
    phase_data.iloc[index_loc, start_loc] = pd.to_datetime(
        "2020-10-22 10:33:14.340000+0200"
    )

    return phase_data


def phase_219(phase_data: pd.DataFrame) -> pd.DataFrame:
    """
    Phase 1.1: U-turn directly before starting, need to remove from phase (true start: 10:03:56)
    """

    phase_data.loc[1, "start"] = pd.to_datetime("2021-04-29 10:03:54.780+0200")

    return phase_data


def phase_220(phase_data: pd.DataFrame) -> pd.DataFrame:
    """
    Neither CAN dump nor CAN logger did work during the last 180s of phase 3, scenario rural;
    hence we cut this part out.
    """

    end_loc = phase_data.columns.get_loc("end")
    index_loc = phase_data[
        (phase_data["phase"] == 3) & (phase_data["scenario"] == "rural")
    ].index.item()
    phase_data.iloc[index_loc, end_loc] = pd.to_datetime(
        "2021-04-30 11:25:00.040000+0200"
    )

    return phase_data


def phase_222(phase_data: pd.DataFrame) -> pd.DataFrame:
    """
    Participant 222 took the wrong turn, therefore we cut the driving data off at this point.
    """

    end_loc = phase_data.columns.get_loc("end")
    index_loc = phase_data[
        (phase_data["phase"] == 2) & (phase_data["scenario"] == "city")
    ].index.item()
    phase_data.iloc[index_loc, end_loc] = pd.to_datetime(
        "2021-05-18 10:52:06.000000+0200"
    )

    return phase_data


def intervention_201(gps_data, phase_data, subject):
    """
    GPS data of phase 2 missing, set new end point of phase 2 city (intervention scenario) manually.
    """

    phase_data = remove_intervention(gps_data, phase_data, subject)
    end_loc = phase_data.columns.get_loc("end")
    index_loc = phase_data[
        (phase_data["phase"] == 2) & (phase_data["scenario"] == "city")
    ].index.item()
    phase_data.iloc[index_loc, end_loc] = pd.to_datetime(
        "2020-10-09 11:58:12.000000+0200"
    )
    return phase_data


def intervention_204(gps_data, phase_data, subject):
    """
    GPS data of phase 1 missing, set new end point of phase 1 highway (intervention scenario) manually.
    """

    phase_data = remove_intervention(gps_data, phase_data, subject)
    end_loc = phase_data.columns.get_loc("end")
    index_loc = phase_data[
        (phase_data["phase"] == 1) & (phase_data["scenario"] == "highway")
    ].index.item()
    phase_data.iloc[index_loc, end_loc] = pd.to_datetime(
        "2020-10-22 10:50:05.000000+0200"
    )
    return phase_data


def intervention_205(gps_data, phase_data, subject):
    """
    GPS data of phase 4 missing, set new end point of phase 4 highway (intervention scenario) manually.
    """

    phase_data = remove_intervention(gps_data, phase_data, subject)
    end_loc = phase_data.columns.get_loc("end")
    index_loc = phase_data[
        (phase_data["phase"] == 4) & (phase_data["scenario"] == "highway")
    ].index.item()
    phase_data.iloc[index_loc, end_loc] = pd.to_datetime(
        "2020-10-23 13:24:45.000000+0200"
    )
    return phase_data


def intervention_224(gps_data, phase_data, subject):
    """
    GPS data of phase 2 missing, set new end point of phase 2 highway (intervention scenario) manually.
    """

    phase_data = remove_intervention(gps_data, phase_data, subject)
    end_loc = phase_data.columns.get_loc("end")
    index_loc = phase_data[
        (phase_data["phase"] == 2) & (phase_data["scenario"] == "highway")
    ].index.item()
    phase_data.iloc[index_loc, end_loc] = pd.to_datetime(
        "2021-05-21 11:03:27.000000+0200"
    )
    return phase_data


CUSTOM_CAN = {
    202: can_202,
    211: can_211,
    214: can_214,
}

CUSTOM_PHASES = {204: phase_204, 219: phase_219, 220: phase_220, 222: phase_222}

CUSTOM_INTERVENTION = {
    201: intervention_201,
    204: intervention_204,
    205: intervention_205,
    224: intervention_224,
}
