import time

import numpy as np

from highway_env.road.generation.generator import default_params, generate_random_lanes
from highway_env.road.lane import PolyLane
from highway_env.road.partitioned_road import PartitionedRoadNetwork
from highway_env.road.road import LineType, RoadNetwork


TIME_TRIALS = 100


def test_partitioned_road_network():
    generation_params = default_params()
    generation_params["disable_prints"] = True

    target_num_endpoints = 2
    while target_num_endpoints <= 256:
        generation_params["target_num_endpoints"] = target_num_endpoints
        print(f"\ntarget_num_endpoints = {target_num_endpoints}")
        lanes = generate_random_lanes(np.random.default_rng(0), generation_params)

        avg_elapsed_regular, avg_elapsed_partitioned = assess_get_closest_index(lanes)
        print("RoadNetwork v. PartitionedRoadNetwork: get_closest_lane_index")
        print(
            f"RoadNetwork avg elapsed time:\t{round(1000 * avg_elapsed_regular, 2)} ms"
        )
        print(
            f"PartitionedRoadNetwork avg elapsed time:\t{round(1000 * avg_elapsed_partitioned, 2)} ms"
        )
        # Timings are diagnostic: tiny graphs and shared CI runners can make
        # partitioning slower. Correctness is checked for every query below.

        target_num_endpoints *= 2


def assess_get_closest_index(lanes):
    partitioned_net = create_partitioned_network(lanes)
    regular_net = create_regular_network(lanes)

    rng = np.random.default_rng(0)
    avg_elapsed_regular = 0
    avg_elapsed_partitioned = 0
    for _ in range(TIME_TRIALS):
        # Choosing a random centerpoint of a random lane
        position = rng.choice(rng.choice(lanes).points)
        heading = 0

        start_time = time.perf_counter()
        regular_index = regular_net.get_closest_lane_index(position, heading)
        elapsed = time.perf_counter() - start_time
        avg_elapsed_regular += elapsed / TIME_TRIALS

        start_time = time.perf_counter()
        partitioned_index = partitioned_net.get_closest_lane_index(position, heading)
        elapsed = time.perf_counter() - start_time
        avg_elapsed_partitioned += elapsed / TIME_TRIALS

        # Equal-distance ties may return different lanes; both searches must
        # still find the same minimum distance, including heading penalties.
        np.testing.assert_allclose(
            partitioned_net.get_lane(partitioned_index).distance_with_heading(
                position, heading
            ),
            regular_net.get_lane(regular_index).distance_with_heading(
                position, heading
            ),
        )

    return avg_elapsed_regular, avg_elapsed_partitioned


def create_partitioned_network(lanes) -> PartitionedRoadNetwork:
    net = PartitionedRoadNetwork(partition_gridsize=100)

    for lane in lanes:
        real_lane = PolyLane(
            lane_points=lane.points,
            left_boundary_points=lane.left_points,
            right_boundary_points=lane.right_points,
            line_types=(LineType.CONTINUOUS, LineType.CONTINUOUS),
        )
        net.add_lane_bidirectional(lane.start, lane.end, real_lane)

    return net


def create_regular_network(lanes) -> RoadNetwork:
    net = RoadNetwork()

    for lane in lanes:
        real_lane = PolyLane(
            lane_points=lane.points,
            left_boundary_points=lane.left_points,
            right_boundary_points=lane.right_points,
            line_types=(LineType.CONTINUOUS, LineType.CONTINUOUS),
        )
        net.add_lane_bidirectional(lane.start, lane.end, real_lane)

    return net
