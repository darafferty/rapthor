#!/usr/bin/env python3

import os
import pandas as pd
import textwrap


def to_etrs_string(df):
    return df.to_csv(
        columns=["ETRS-X", "ETRS-Y", "ETRS-Z"],
        index=False,
        header=False,
        float_format="%.3f",
        sep=",",
    )


def check_file_exists(filename):
    if not os.path.isfile(filename):
        print(
            f"{filename} not found, you can get it from https://github.com/lofar-astron/lofar-antenna-positions/tree/master/share/lofarantpos"
        )
        return False
    return True


def print_stations(df):
    station_names = " ".join(df["STATION"].unique())
    station_names = textwrap.wrap(station_names, 80)
    print("\n".join(station_names))
    print()


def write_output(outputs):
    for filename, df in outputs.items():
        print(f"Writing {len(df)} coordinates to {filename}")
        open(filename, "w").write(to_etrs_string(df))


def parse_lofar():
    input_filename = "etrs-phase-centres.csv"
    if not check_file_exists(input_filename):
        return

    df = pd.read_csv(input_filename)
    print("LOFAR stations:")
    print_stations(df)

    df_hba = df.query("FIELD == 'HBA'")
    df_lba = df.query("FIELD == 'LBA'")

    write_output({"LOFAR_hba.txt": df_hba, "LOFAR_lba.txt": df_lba})
    print()


def parse_aartfaac():
    antenna_sets = {"A6": "CS002 CS003 CS004 CS005 CS006 CS007".split()}
    antenna_sets["A12"] = (
        antenna_sets["A6"] + "CS001 CS011 CS013 CS017 CS021 CS032".split()
    )
    antenna_sets["A24"] = (
        antenna_sets["A12"]
        + "CS024 CS026 CS028 CS030 CS031 CS101 CS103 CS201 CS301 CS302 CS401 CS501".split()
    )
    antenna_sets

    input_filename = "etrs-antenna-positions.csv"
    if not check_file_exists(input_filename):
        return
    df = pd.read_csv(input_filename)
    df_lba = df.query("`ANTENNA-TYPE` == 'LBA'")

    nr_antennas_per_station = 48
    df_lba_outer = pd.concat(
        [
            df_lba.iloc[i : i + nr_antennas_per_station]
            for i in range(0, len(df_lba), nr_antennas_per_station)
        ],
        ignore_index=True,
    )

    df_a6 = df_lba_outer[df_lba_outer["STATION"].isin(antenna_sets["A6"])]
    df_a12 = df_lba_outer[df_lba_outer["STATION"].isin(antenna_sets["A12"])]
    df_a24 = df_lba_outer[df_lba_outer["STATION"].isin(antenna_sets["A24"])]

    print("AARTFAAC-6 stations:")
    print_stations(df_a6)

    print("AARTFAAC-12 stations:")
    print_stations(df_a12)

    print("AARTFAAC-24 stations:")
    print_stations(df_a24)

    write_output(
        {
            "AARTFAAC_6.txt": df_a6,
            "AARTFAAC_12.txt": df_a12,
            "AARTFAAC_24.txt": df_a24,
        }
    )


if __name__ == "__main__":
    parse_lofar()
    parse_aartfaac()
