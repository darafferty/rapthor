import numpy as np

baselinetype = np.dtype([("station1", np.intc), ("station2", np.intc)])

coordinatetype = np.dtype([("x", np.intc), ("y", np.intc), ("z", np.intc)])

metadatatype = np.dtype(
    [
        ("time_index", np.intc),
        ("nr_timesteps", np.intc),
        ("channel_begin", np.intc),
        ("channel_end", np.intc),
        ("baseline", baselinetype),
        ("coordinate", coordinatetype),
        ("wtile_coordinate", coordinatetype),
        ("wtile_index", np.intc),
        ("nr_aterms", np.intc),
    ]
)


def get_metadata(
    nr_subgrids, nr_baselines, nr_timeslots, nr_timesteps_per_subgrid, nr_channels
):
    metadata = np.zeros(nr_subgrids, dtype=metadatatype)

    for bl in range(nr_baselines):
        for ts in range(nr_timeslots):
            idx = bl * nr_timeslots + ts
            m = metadata[idx]
            m["time_index"] = ts * nr_timesteps_per_subgrid
            m["nr_timesteps"] = nr_timesteps_per_subgrid
            m["channel_begin"] = 0
            m["channel_end"] = nr_channels
            m["baseline"] = bl
            m["coordinate"] = (0, 0, 0)
            m["wtile_coordinate"] = (0, 0, 0)
            m["nr_aterms"] = 1

    return metadata


def get_aterms_indices(nr_timesteps_per_baseline, nr_timesteps_per_subgrid):
    aterms_indices = np.zeros((nr_timesteps_per_baseline), dtype=np.int32)
    for t in range(nr_timesteps_per_baseline):
        aterms_indices[t] = np.floor(t / nr_timesteps_per_subgrid)

    return aterms_indices


def flops_gridder(
    nr_channels, nr_timesteps, nr_subgrids, subgrid_size, nr_correlations
):

    # Number of flops per visibility
    flops_per_visibility = 0
    flops_per_visibility += 5  # phase index
    flops_per_visibility += 5  # phase offset
    flops_per_visibility += nr_channels * 2  # phase
    flops_per_visibility += nr_channels * nr_correlations * 8  # update

    # Number of flops per subgrid
    flops_per_subgrid = 6  # shift

    # Total number of flops
    flops_total = 0
    flops_total += nr_timesteps * subgrid_size * subgrid_size * flops_per_visibility
    flops_total += nr_subgrids * subgrid_size * subgrid_size * flops_per_subgrid

    return flops_total
