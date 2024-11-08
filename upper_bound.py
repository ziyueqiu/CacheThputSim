import math
import os

import numpy as np
import matplotlib.pyplot as plt

# Set larger font sizes
plt.rcParams.update({'font.size': 32})  # Adjust font size as needed


# Define the function for X
def calculate_X_LRU(p_hit, N=72, disk_latency=100):
    S_tail = 0 # S_tail = 0 ~ 0.59
    D = 0.7 * p_hit + S_tail * (1 - p_hit) + 0.59
    Z = 0.51 + (1 - p_hit) * disk_latency
    D_max = max(0.59, 0.7 * p_hit)
    part1 = N / (D + Z)
    part2 = 1 / D_max
    return min(part1, part2)


def calculate_X_FIFO(p_hit, N=72, disk_latency=100):
    S_tail = 0  # S_tail = 0 ~ 0.73
    D = (0.73 + S_tail) * (1 - p_hit)
    Z = 0.51 + (1 - p_hit) * disk_latency
    D_max = 0.67 * (1 - p_hit)
    part1 = N / (D + Z)
    if D_max == 0:
        return part1
    part2 = 1 / D_max
    return min(part1, part2)


def calculate_X_probLRU_2(p_hit, N=72, disk_latency=100, q=0.54):
    S_tail = 0 # < 0.65
    D = (1-q) * p_hit * 0.78 + (1-p_hit) * S_tail + (1-p_hit+(1-q)*p_hit) * 0.65
    Z = 0.51 + (1 - p_hit) * disk_latency
    D_max = max(0.78 * p_hit * (1-q), (p_hit * (1-q) + 1-p_hit) * 0.65)
    # print(D_max)
    part1 = N / (D + Z)
    if D_max == 0:
        return part1
    part2 = 1 / D_max
    return min(part1, part2)

# print(calculate_X_probLRU_2(0.8, N=72, disk_latency=5))
def calculate_X_probLRU_72(p_hit, N=72, disk_latency=100):
    S_tail = 0 # < 0.67
    q = 1 - 1/72
    D = (1-q) * p_hit * 0.81 + (1-p_hit) * S_tail + (1-p_hit+(1-q)*p_hit) * 0.67
    Z = 0.51 + (1 - p_hit) * disk_latency
    D_max = max(0.81 * p_hit * (1-q), (p_hit * (1-q) + 1-p_hit) * 0.67)
    part1 = N / (D + Z)
    if D_max == 0:
        return part1
    part2 = 1 / D_max
    return min(part1, part2)

def calculate_X_CLOCK(p_hit, N=72, disk_latency=100):
    S_tail = 0  # < 0.65
    g = 2.43 * 10**-5 * math.exp(11.24 * p_hit) + 0.187
    Z = 0.51 + (1 - p_hit) * disk_latency
    D_max = (0.65 + 0.3 * g) * (1 - p_hit)
    D = S_tail * (1 - p_hit) + D_max
    part1 = N / (D + Z)
    if D_max == 0:
        return part1
    part2 = 1 / D_max
    return min(part1, part2)

def calculate_X_SLRU(p_hit, N=72, disk_latency=100):
    S_tail = 0  # < 0.59
    p_hit_protected = -0.1158 * p_hit**2 + 1.0112 * p_hit - 0.0009
    p_hit_prob = p_hit - p_hit_protected
    assert p_hit_protected >= p_hit_prob
    p_miss = 1 - p_hit
    D_max = max(0.7 * p_hit_protected, # Delink High
                0.59 * (p_hit_prob + p_miss), # Head Low
                0.59 * p_hit, # Head High
                )
    Z = 0.51 + (1 - p_hit) * disk_latency
    D = 0.7 * p_hit \
        + S_tail * (p_hit_prob + p_miss) \
        + 0.59 * (p_hit_prob + p_miss) \
        + 0.59 * p_hit
    return min(N/(D+Z), 1/D_max)

def calculate_X_S3FIFO(p_hit, p1, p2, N=72, disk_latency=100):
    g = 2.43 * 10 ** -5 * math.exp(11.24 * p_hit) + 0.187
    Z = 0.51 + (1 - p_hit) * (disk_latency + 0.51)
    p_miss = 1 - p_hit
    p_ghost = p1 * p_miss
    p_M = p2 * (p_miss - p_ghost)
    D_head_M = (p_M + p_ghost) * 0.65
    D_S = 0.65 * 2 * (p_miss - p_ghost)
    D_tail_M = (p_M + p_ghost) * (0.65 + 0.3 * g)
    D = D_head_M + D_S + D_tail_M
    D_max = max(0.65 * (p_miss - p_ghost), D_tail_M)
    if D_max == 0:
        return N/(D+Z)
    return min(N/(D+Z), 1/D_max)

def draw(dir_name="output"):
    for algo in ["LRU", "FIFO", "probLRU_2", "probLRU_72"]:
        for disk_latency in [500, 100, 5]:
            # Generate p_hit values
            p_hit_values = np.linspace(0.4, 1, 100)  # p_hit ranging from 0.4 to 1
            # Plotting
            plt.figure(figsize=(12, 12))

            # MPL_values = [72, 144]
            MPL_values = [72]
            for MPL in MPL_values:
                # Calculate X for each p_hit
                if algo == "LRU":
                    plt.ylim(0, 2)
                    X_values = [calculate_X_LRU(p_hit, disk_latency=disk_latency, N=MPL) for p_hit in p_hit_values]
                elif algo == "FIFO":
                    plt.ylim(0, 200)
                    X_values = [calculate_X_FIFO(p_hit, disk_latency=disk_latency, N=MPL) for p_hit in p_hit_values]
                elif algo == "probLRU_2":
                    plt.ylim(0, 4)
                    X_values = [calculate_X_probLRU_2(p_hit, disk_latency=disk_latency, N=MPL) for p_hit in p_hit_values]
                elif algo == "probLRU_72":
                    plt.ylim(0, 100)
                    X_values = [calculate_X_probLRU_72(p_hit, disk_latency=disk_latency, N=MPL) for p_hit in p_hit_values]
                else:
                    assert False

                # Plotting
                plt.plot(p_hit_values, X_values, label=f'MPL={MPL}', linewidth=8)
                plt.xlabel('Hit Ratio $p_{\mathrm{hit}}$', fontsize=40)
                plt.ylabel('Throughput X', fontsize=40)

                # plt.title('Disk Latency={}us'.format(disk_latency))

            if len(MPL_values) > 1:
                plt.legend()

            plt.grid(True)
            plt.tight_layout()  # Adjust layout
            if len(MPL_values) > 1:
                plt.savefig(os.path.join(dir_name, 'discussion_{}_{}.eps'.format(algo, disk_latency)))
            else:
                plt.savefig(os.path.join(dir_name, 'x_bound_{}_{}.png'.format(algo, disk_latency)))
            plt.close()


# draw()
