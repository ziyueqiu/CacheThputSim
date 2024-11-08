import csv
import math
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from enum import Enum, auto
from abc import ABC, abstractmethod

from scipy.stats import chi2


class Stat:
    def __init__(self):
        self.E_X = 0
        self.E_X2 = 0
        self.N = 0

    def Add(self, x):
        self.E_X += x
        self.E_X2 += x * x
        self.N += 1

    def GetMean(self):
        if self.N == 0:
            return 0
        else:
            return self.E_X / self.N

    def GetCount(self):
        return self.N


class NewStat:
    def __init__(self):
        self.E_X = 0
        self.T = 0

    def Add(self, x, t):
        if t == 0:
            return
        self.E_X = self.E_X * (self.T / (self.T + t)) + x * t / (self.T + t)
        self.T += t

    def GetMean(self):
        return self.E_X


class FCFSQueue:
    def __init__(self, job_size_mean=None):
        self.queue = []
        self.job_size_mean = job_size_mean

    def push(self, job, append=True):
        self.queue.append(job)
        if not append:
            # sort by time_arrival
            self.queue.sort(key=lambda x: x.get_time_arrival())

    def pop(self):
        return self.queue.pop(0)

    def peek(self):
        return self.queue[0] if len(self.queue) > 0 else None

    def is_empty(self):
        return len(self.queue) == 0

    def show(self):
        if self.is_empty():
            print("Empty")
        else:
            for job in self.queue:
                print(f"({job.get_time_arrival():.3f}, {job.get_job_size():.3f})", end=" ")
            print()


class Job:
    def __init__(self, time_arrival, job_size):
        self.time_arrival = time_arrival
        self.job_size = job_size

    def get_time_arrival(self):
        return self.time_arrival

    def get_job_size(self):
        return self.job_size

    def run_job(self, time_duration):
        self.job_size -= time_duration


class SingleLRUSimulator:
    def __init__(self, hit_ratio, MPL, disk_lat, job_size_mean_list=None):
        self.QUEUE = FCFSQueue()
        self.t = 0
        self.rho = NewStat()
        self.E_T = Stat()  # queueing time for each queue w/o think time
        self.E_S = Stat()
        self.req_num = 0
        self.MPL = MPL

        if job_size_mean_list is None:
            job_size_mean_list = [0.5, 0.9, 0.95]
        else:
            assert len(job_size_mean_list) == 3  # hash think time, hit, miss

        self.hash_job_size_mean = job_size_mean_list[0]
        self.hit_job_size_mean = job_size_mean_list[1]
        self.miss_job_size_mean = job_size_mean_list[2]

        self.hit_ratio = hit_ratio
        self.disk_lat = disk_lat

    def hit_ratio_condition(self, ratio):
        if random.random() < ratio:
            return True
        else:
            return False

    def push_to_queue(self, time_arrival, job_size_mean):
        if job_size_mean == 0:
            job_size = 0
        else:
            job_size = np.random.exponential(job_size_mean)
        self.QUEUE.push(Job(time_arrival, job_size), append=False)
        self.E_S.Add(job_size)

    def RunQueue(self, time_duration):
        flag = False
        assert not self.QUEUE.is_empty()
        # if self.QUEUE.is_empty():
        #     self.rho.Add(0, time_duration)
        #     return flag
        if self.QUEUE.peek().get_time_arrival() < self.t:
            self.QUEUE.peek().run_job(time_duration)
            self.rho.Add(1, time_duration)
            if self.QUEUE.peek().get_job_size() <= 0:
                self.E_T.Add(self.t - self.QUEUE.peek().get_time_arrival())
                self.QUEUE.pop()
        elif self.QUEUE.peek().get_time_arrival() == self.t:  # Job size == 0
            if self.QUEUE.peek().get_job_size() == 0:
                self.E_T.Add(self.t - self.QUEUE.peek().get_time_arrival())
                self.QUEUE.pop()
                flag = True
        else:
            self.rho.Add(0, time_duration)
        return flag

    def GetNextEvent(self):
        assert not self.QUEUE.is_empty()
        if self.QUEUE.peek().get_time_arrival() <= self.t:
            return self.QUEUE.peek().get_job_size(), True  # Job completion
        else:
            return self.QUEUE.peek().get_time_arrival() - self.t, False

    def start(self):
        if self.hit_ratio_condition(self.hit_ratio):
            self.push_to_queue(self.t + self.hash_job_size_mean, self.hit_job_size_mean)
        else:
            self.push_to_queue(self.t + self.hash_job_size_mean + self.disk_lat, self.miss_job_size_mean)

    def stat(self):
        print("t:", self.t)
        print("rho:", self.rho.GetMean())
        print("E_T:", self.E_T.GetMean())
        print("E_S:", self.E_S.GetMean())
        if self.req_num != 0:
            print(f"X KIOPS: {self.req_num / self.t * 1000:.0f}")

    def program(self, number_of_req):
        while self.req_num < number_of_req:
            next_time, is_completion = self.GetNextEvent()
            self.t += next_time
            self.RunQueue(next_time)
            if is_completion:  # Job completion
                self.req_num += 1
                self.start()

    def run(self, number_of_req):
        for _ in range(self.MPL):
            self.start()
        self.program(number_of_req)
        self.stat()
        return self.req_num / self.t * 1000


# SingleLRUSimulator(0.996, 72, 5).run(100000)  # 1082
# SingleLRUSimulator(0.456, 72, 100, [0.5, 0.9, 0.95]).run(100000)  # 1033

class BaseClosedFCFSSimulator(ABC):
    def __init__(self, number_of_queues, job_size_mean_list):
        assert number_of_queues == len(job_size_mean_list)
        self.number_of_queues = number_of_queues
        # self.job_size_mean_list = job_size_mean_list
        self.QUEUES = {
            i: FCFSQueue(job_size_mean_list[i]) for i in range(number_of_queues)
        }
        self.t = 0
        # self.state = -1
        self.rho = [NewStat() for i in range(number_of_queues)]
        self.E_T = [Stat() for i in range(number_of_queues)]  # queueing time for each queue w/o think time
        self.E_S = [Stat() for i in range(number_of_queues)]
        self.req_num = 0
        self.MPL = None

    def push_to_queue(self, queue_id, time_arrival, append=True):
        job_size_mean = self.QUEUES[queue_id].job_size_mean
        if job_size_mean == 0:
            job_size = 0
        else:
            job_size = np.random.exponential(job_size_mean)
        self.QUEUES[queue_id].push(Job(time_arrival, job_size), append=append)
        self.E_S[queue_id].Add(job_size)

    def RunQueues(self, time_duration):
        flag = False
        for queue_id in range(self.number_of_queues):
            if self.QUEUES[queue_id].is_empty():
                self.rho[queue_id].Add(0, time_duration)
                continue
            if self.QUEUES[queue_id].peek().get_time_arrival() < self.t:
                self.QUEUES[queue_id].peek().run_job(time_duration)
                self.rho[queue_id].Add(1, time_duration)
                if self.QUEUES[queue_id].peek().get_job_size() <= 0:
                    self.E_T[queue_id].Add(self.t - self.QUEUES[queue_id].peek().get_time_arrival())
                    self.QUEUES[queue_id].pop()
            elif self.QUEUES[queue_id].peek().get_time_arrival() == self.t:  # Job size == 0
                if self.QUEUES[queue_id].peek().get_job_size() == 0:
                    self.E_T[queue_id].Add(self.t - self.QUEUES[queue_id].peek().get_time_arrival())  # add 0
                    self.QUEUES[queue_id].pop()
                    flag = True
            else:
                self.rho[queue_id].Add(0, time_duration)
        return flag

    def GetNextEvent(self):
        queue_events = {}
        for queue_id, queue in self.QUEUES.items():
            if not queue.is_empty() and queue.peek().get_time_arrival() <= self.t:
                queue_events[queue_id] = (queue.peek().get_job_size(), True)  # Job completion
            elif not queue.is_empty() and queue.peek().get_time_arrival() > self.t:
                queue_events[queue_id] = (queue.peek().get_time_arrival() - self.t, False)  # Job arrival

        next_time = min(queue_events.values(), key=lambda x: x[0])[0]
        event_queues = [k for k, v in queue_events.items() if v[0] == next_time]
        return event_queues[0], next_time, queue_events[event_queues[0]][1]  # queue_id, next_time, is_completion

    def show(self):
        print("Time: ", self.t, "Req: ", self.req_num)
        for queue_id, queue in self.QUEUES.items():
            print(f"Queue {queue_id} ({len(queue.queue)}): ")
            queue.show()
        self.stat()
        total = sum([len(queue.queue) for queue in self.QUEUES.values()])
        assert total == self.MPL
        print()

    def stat(self):
        print("t:", self.t)
        print("rho:", [rho.GetMean() for rho in self.rho])
        print("E_T:", [E_T.GetMean() for E_T in self.E_T])
        print("E_T count:", [E_T.GetCount() for E_T in self.E_T])
        print("E_S:", [E_S.GetMean() for E_S in self.E_S])
        if self.req_num != 0:
            print(f"X KIOPS: {self.req_num / self.t * 1000:.0f}")

    def run(self, number_of_req):
        for _ in range(self.MPL):
            self.start()
        self.program(number_of_req)
        self.stat()
        return self.req_num / self.t * 1000

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def program(self, number_of_req):
        pass


class LRUSimulator(BaseClosedFCFSSimulator):
    class Type(Enum):
        DELINK = 0
        TAIL = 1
        HEAD = 2

    def __init__(self, hit_ratio, MPL, disk_lat):
        number_of_queues = 3
        # debugging: 0.9, 0.64, 0.245
        job_size_mean_list = [0.7, 0.59, 0.59]
        # job_size_mean_list = [0.9, 0.4, 0.4]
        super().__init__(number_of_queues, job_size_mean_list)
        self.hash_job_size_mean = 0.51
        self.hit_ratio = hit_ratio
        self.MPL = MPL
        self.disk_lat = disk_lat

    def hit_ratio_condition(self, ratio):
        if random.random() < ratio:
            return True
        else:
            return False

    # all goes over think time, then hit to DELINK, miss to TAIL, then all goes to HEAD
    def start(self):
        if self.hit_ratio_condition(self.hit_ratio):
            self.push_to_queue(self.Type.DELINK.value, self.t + self.hash_job_size_mean)
        else:
            self.push_to_queue(self.Type.TAIL.value, self.t + self.hash_job_size_mean + self.disk_lat)

    def program(self, number_of_req):
        while self.req_num < number_of_req:
            queue_id, next_time, is_completion = self.GetNextEvent()
            # print(queue_id, next_time, is_completion)
            self.t += next_time
            if not is_completion:  # Job arrival
                self.RunQueues(next_time)
            else:  # Job completion
                self.RunQueues(next_time)
                if queue_id == self.Type.HEAD.value:
                    self.req_num += 1
                    self.start()
                else:
                    self.push_to_queue(self.Type.HEAD.value, self.t)


# LRUSimulator(0.997, 72, 5).run(100000)  # 1436
# LRUSimulator(0.98, 72, 5).run(100000)  # 1475
# LRUSimulator(0.897, 72, 5).run(100000)  # 1601
# LRUSimulator(0.5879, 72, 5).run(100000)  # 1787
# LRUSimulator(0.46, 72, 5).run(100000)  # 1757
# LRUSimulator(0.459, 72, 100).run(100000)  # 1256
# LRUSimulator(0.967, 72, 500).run(100000)  # 1443
# LRUSimulator(0.7597, 72, 100).run(100000)  # 1617
# LRUSimulator(0, 72, 5).run(100000)  # 1690


class SLRUSimulator(BaseClosedFCFSSimulator):
    class Type(Enum):
        DELINK_H = 0
        DELINK_L = 1
        TAIL_L = 2
        HEAD_L = 3
        HEAD_H = 4

    class SegMent(Enum):
        SEGMENT_H = 0
        SEGMENT_L = 1
        OTHER = 2

    def __init__(self, hit_ratio_H, hit_ratio_L, MPL, disk_lat):
        number_of_queues = 5
        job_size_mean_list = [0.7, 0.7, 0.59, 0.59, 0.59]
        # job_size_mean_list = [0.9, 0.9, 0.3, 0.245, 0.245]
        super().__init__(number_of_queues, job_size_mean_list)
        self.hash_job_size_mean = 0.51
        assert hit_ratio_H + hit_ratio_L <= 1
        self.hit_ratio_H = hit_ratio_H
        self.hit_ratio_L = hit_ratio_L
        self.MPL = MPL
        self.disk_lat = disk_lat

    def hit_ratio_condition(self, ratio_H, ratio_L):
        rand = random.random()
        if rand < ratio_H:
            return self.SegMent.SEGMENT_H
        elif rand < ratio_H + ratio_L:
            return self.SegMent.SEGMENT_L
        else:
            return self.SegMent.OTHER

    def start(self):
        condition = self.hit_ratio_condition(self.hit_ratio_H, self.hit_ratio_L)
        if condition == self.SegMent.SEGMENT_H:
            self.push_to_queue(self.Type.DELINK_H.value, self.t + self.hash_job_size_mean, append=False)
        elif condition == self.SegMent.SEGMENT_L:
            self.push_to_queue(self.Type.DELINK_L.value, self.t + self.hash_job_size_mean, append=False)
        else:
            self.push_to_queue(self.Type.TAIL_L.value, self.t + self.hash_job_size_mean + self.disk_lat, append=False)

    def program(self, number_of_req):
        while self.req_num < number_of_req:
            queue_id, next_time, is_completion = self.GetNextEvent()
            # print(queue_id, next_time, is_completion)
            self.t += next_time
            if not is_completion:
                self.RunQueues(next_time)
            else:
                self.RunQueues(next_time)
                if queue_id == self.Type.HEAD_H.value:
                    # or queue_id == self.Type.HEAD_L.value):
                    self.req_num += 1
                    self.start()
                elif queue_id == self.Type.TAIL_L.value or queue_id == self.Type.DELINK_L.value:
                    self.push_to_queue(self.Type.HEAD_L.value, self.t, append=False)
                elif queue_id == self.Type.DELINK_H.value:
                    self.push_to_queue(self.Type.HEAD_H.value, self.t, append=False)
                else:  # HEAD_L
                    rand_ratio = random.random()
                    if rand_ratio < self.hit_ratio_L / (1 - self.hit_ratio_H):
                        self.push_to_queue(self.Type.HEAD_H.value, self.t, append=False)
                    else:
                        self.req_num += 1
                        self.start()


class ProbLRUSimulator(BaseClosedFCFSSimulator):
    class Type(Enum):
        EMPTY = 0
        DELINK = 1
        TAIL = 2
        HEAD = 3

    class SegMent(Enum):
        Ret = 0
        Hit = 1
        Miss = 2

    def __init__(self, hit_ratio, MPL, disk_lat, prob, delink=0.81, tail=0.64, head=0.73):
        number_of_queues = 4
        job_size_mean_list = [0, delink, tail, head]  # default prob=1/72
        super().__init__(number_of_queues, job_size_mean_list)
        self.hash_job_size_mean = 0.51
        self.hit_ratio = hit_ratio
        self.MPL = MPL
        self.disk_lat = disk_lat
        self.prob = prob

    def hit_ratio_condition(self):
        rand = random.random()
        if rand < self.hit_ratio * self.prob:
            return self.SegMent.Ret
        elif rand < self.hit_ratio:
            return self.SegMent.Hit
        else:
            return self.SegMent.Miss

    def start(self):
        condition = self.hit_ratio_condition()
        if condition == self.SegMent.Ret:
            self.push_to_queue(self.Type.EMPTY.value, self.t + self.hash_job_size_mean)
        elif condition == self.SegMent.Hit:
            self.push_to_queue(self.Type.DELINK.value, self.t + self.hash_job_size_mean)
        else:
            self.push_to_queue(self.Type.TAIL.value, self.t + self.hash_job_size_mean + self.disk_lat)

    def program(self, number_of_req):
        while self.req_num < number_of_req:
            queue_id, next_time, is_completion = self.GetNextEvent()
            # print(queue_id, next_time, is_completion)
            self.t += next_time
            if not is_completion:
                ret = self.RunQueues(next_time)
                if ret:
                    self.start()
                    self.req_num += 1
            else:
                self.RunQueues(next_time)
                if queue_id == self.Type.HEAD.value or queue_id == self.Type.EMPTY.value:
                    self.req_num += 1
                    self.start()
                else:  # DELINK or TAIL
                    self.push_to_queue(self.Type.HEAD.value, self.t)
            # self.show()


# ProbLRUSimulator(0.9995, 72, 5, 0.16, delink=0.76, head=0.64).run(100000) # 1569
# ProbLRUSimulator(0.895, 72, 5, 0.1767, delink=0.76, head=0.64).run(100000) # 1689
# ProbLRUSimulator(0.455, 72, 5, 0.256, delink=0.76, head=0.64).run(100000) # 1776
# ProbLRUSimulator(0.9995, 72, 500, 0.157776, delink=0.76, head=0.64).run(100000) # 1514

# ProbLRUSimulator(0.998, 72, 5, 0.53, delink=0.78, tail=0.64, head=0.65).run(100000) # 2708
# ProbLRUSimulator(0.998, 72, 100, 0.53, delink=0.78, tail=0.64, head=0.65).run(100000) # 2641
# ProbLRUSimulator(0.998, 72, 500, 0.5227, delink=0.78, tail=0.64, head=0.65).run(100000) # 2716
# ProbLRUSimulator(0.8, 72, 5, 0.542, delink=0.78, head=0.65).run(100000) # 2685
# ProbLRUSimulator(0.4497689, 72, 5, 0.57, delink=0.78, head=0.65).run(100000) # 2055


# ProbLRUSimulator(1, 72, 5, 71/72, delink=0.81, tail=0.64, head=0.67).run(200000) # 88106
# ProbLRUSimulator(1, 72, 100, 71/72, delink=0.81, tail=0.64, head=0.67).run(200000) # 90555
# ProbLRUSimulator(1, 72, 500, 71/72, delink=0.81, tail=0.64, head=0.67).run(200000) # 89793
# ProbLRUSimulator(0.989550, 72, 0, 71/72).run(100000) # 62678
# ProbLRUSimulator(0.975156, 72, 0, 71/72).run(100000) # 39325
# ProbLRUSimulator(0.903549, 72, 0, 71/72).run(100000) # 13271
# ProbLRUSimulator(0.8, 72, 0, 71/72).run(100000) # >6811
# ProbLRUSimulator(0.416634, 72, 0, 71/72).run(100000) # 2563

class FIFOSimulator(ProbLRUSimulator):
    # prob = 100%
    def __init__(self, hit_ratio, MPL, disk_lat):
        super().__init__(hit_ratio, MPL, disk_lat, 1)


# FIFOSimulator(0.99999, 72, 5).run(100000) # 125396
# FIFOSimulator(0.9866, 72, 5).run(100000) # 88727
# FIFOSimulator(0.9597, 72, 5).run(100000) # 32828
# FIFOSimulator(0.47, 72, 5).run(100000) # 2601


class TProbLRUSimulator(BaseClosedFCFSSimulator):
    class Type(Enum):
        EMPTY = 0
        DELINK = 1
        TAIL = 2
        HEAD = 3

    class SegMent(Enum):
        Ret = 0
        Hit = 1
        Miss = 2

    def __init__(self, hit_ratio, MPL, disk_lat, prob, compare_fail_prob, delink=0.81, tail=0.64, head=0.67,
                 job_size_mean=0.51, insert=0.0):
        number_of_queues = 4
        job_size_mean_list = [0, delink, tail, head]  # default prob=1/72
        super().__init__(number_of_queues, job_size_mean_list)
        self.hash_job_size_mean = job_size_mean
        self.insert = insert
        self.hit_ratio = hit_ratio
        self.MPL = MPL
        self.disk_lat = disk_lat
        self.prob = prob
        self.compare_fail_prob = compare_fail_prob  # if fail, return, not admitted to head

    def hit_ratio_condition(self):
        rand = random.random()
        if rand < self.hit_ratio * self.prob:
            return self.SegMent.Ret
        elif rand < self.hit_ratio:
            return self.SegMent.Hit
        else:
            return self.SegMent.Miss

    def compare_fail_condition(self):
        if random.random() < self.compare_fail_prob:
            return True
        else:
            return False

    def start(self):
        condition = self.hit_ratio_condition()
        if condition == self.SegMent.Ret:
            self.push_to_queue(self.Type.EMPTY.value, self.t + self.hash_job_size_mean, append=False)
        elif condition == self.SegMent.Hit:
            self.push_to_queue(self.Type.DELINK.value, self.t + self.hash_job_size_mean)
        else:
            if self.compare_fail_condition():
                self.push_to_queue(self.Type.EMPTY.value,
                                   self.t + self.hash_job_size_mean + self.disk_lat + self.insert, append=False)
            else:
                self.push_to_queue(self.Type.TAIL.value,
                                   self.t + self.hash_job_size_mean + self.disk_lat + self.insert)  # CBF add + check

    def program(self, number_of_req):
        while self.req_num < number_of_req:
            queue_id, next_time, is_completion = self.GetNextEvent()
            # print(queue_id, next_time, is_completion)
            self.t += next_time
            if not is_completion:
                ret = self.RunQueues(next_time)
                if ret:
                    self.start()
                    self.req_num += 1
            else:
                self.RunQueues(next_time)
                if queue_id == self.Type.HEAD.value or queue_id == self.Type.EMPTY.value:
                    self.req_num += 1
                    self.start()
                else:  # DELINK or TAIL
                    self.push_to_queue(self.Type.HEAD.value, self.t)
            # self.show()


# TFIFO
# TProbLRUSimulator(0.467, 72, 7, 1, 0.922,
#                   tail=0.64, head=0.552, job_size_mean=3.684).run(100000) # 7836
# TProbLRUSimulator(0.589, 72, 4.7, 1, 0.967,
#                   tail=0.64, head=0.552, job_size_mean=4.684).run(100000) # 8828

# TLRU
# TLRU,5,0.0005,0.4863013698630137,3139.0,22.93724115960497,72,2,0.7889918226520688,,
# TProbLRUSimulator(0.486, 72, 5, 0, 0.79, delink=0.68, head=0.55).run(100000) # 3139
# TProbLRUSimulator(0.62, 72, 5, 0, 0.77, delink=0.68, head=0.55).run(100000) # 2586
# TProbLRUSimulator(0.9, 72, 5, 0, 0.53, delink=0.68, head=0.55).run(100000) # 1630
# TProbLRUSimulator(0.997, 72, 5, 0, 0.0, delink=0.68, head=0.55).run(100000) # 1454

class S3FIFOSimulator(BaseClosedFCFSSimulator):
    class Type(Enum):
        HEAD_S = 0
        HEAD_M = 1
        TAIL_S = 2
        TAIL_M = 3
        EMPTY = 4

    def __init__(self, hit_ratio, p1, p2, x3, MPL, disk_lat):
        number_of_queues = 5
        job_size_mean_list = [0.65, 0.65, 0.65, 0.65 + 0.3 * x3, 0]
        super().__init__(number_of_queues, job_size_mean_list)
        self.hash_job_size_mean = 0.51
        assert p1 <= 1
        assert p2 <= 1
        self.p1 = p1
        self.p2 = p2
        print("p1", self.p1, "p2", self.p2)
        self.MPL = MPL
        self.disk_lat = disk_lat
        self.hit_ratio = hit_ratio

    def hit_ratio_condition(self):
        rand = random.random()
        if rand < self.hit_ratio:
            return True
        else:
            return False

    def start(self):
        condition = self.hit_ratio_condition()
        if condition:
            self.push_to_queue(self.Type.EMPTY.value, self.t + self.hash_job_size_mean)
        else:  # miss
            if random.random() < self.p1:
                self.push_to_queue(self.Type.HEAD_M.value, self.t + self.hash_job_size_mean * 2 + self.disk_lat)
            else:
                self.push_to_queue(self.Type.HEAD_S.value, self.t + self.hash_job_size_mean * 2 + self.disk_lat)

    def program(self, number_of_req):
        while self.req_num < number_of_req:
            queue_id, next_time, is_completion = self.GetNextEvent()
            # if queue_id != self.Type.EMPTY.value:
            #     print(queue_id, next_time, is_completion)
            #     self.show()
            self.t += next_time
            if not is_completion:
                ret = self.RunQueues(next_time)
                if ret:
                    self.start()
                    self.req_num += 1
                # self.RunQueues(next_time)
            else:
                self.RunQueues(next_time)
                if queue_id == self.Type.EMPTY.value or queue_id == self.Type.TAIL_M.value:
                    self.req_num += 1
                    self.start()
                elif queue_id == self.Type.HEAD_M.value:
                    self.push_to_queue(self.Type.TAIL_M.value, self.t)
                elif queue_id == self.Type.HEAD_S.value:
                    self.push_to_queue(self.Type.TAIL_S.value, self.t)
                else:  # TAIL_S
                    if random.random() < self.p2:
                        self.push_to_queue(self.Type.HEAD_M.value, self.t)
                    else:
                        self.req_num += 1
                        self.start()


# print("p1", chi2.pdf(0.05 * 65, 4.4912, 1.1394, 3.595) / 0.05)
# print("p2", chi2.pdf(0.05 * 400, 2.2870, 4.5309, 26.5874) / 0.05)
# print("x3", 2.43 * 10 ** -5 * math.exp(11.24 * 0.95) + 0.187)
# S3FIFOSimulator(0.95, chi2.pdf(0.05 * 65, 4.4912, 1.1394, 3.595) / 0.05,
#                 chi2.pdf(0.05 * 400, 2.2870, 4.5309, 26.5874) / 0.05,
#                 2.43 * 10 ** -5 * math.exp(11.24 * 0.95) + 0.187, 72, 5).run(100000)
#
# print("p1", chi2.pdf(0.04 * 65, 4.4912, 1.1394, 3.595) / 0.04)
# print("p2", chi2.pdf(0.04 * 400, 2.2870, 4.5309, 26.5874) / 0.04)
# print("x3", 2.43 * 10 ** -5 * math.exp(11.24 * 0.96) + 0.187)
# S3FIFOSimulator(0.96, chi2.pdf(0.04 * 65, 4.4912, 1.1394, 3.595) / 0.04,
#                 chi2.pdf(0.04 * 400, 2.2870, 4.5309, 26.5874) / 0.04,
#                 2.43 * 10 ** -5 * math.exp(11.24 * 0.96) + 0.187, 72, 5).run(100000)
#
# print("p1", chi2.pdf(0.03 * 65, 4.4912, 1.1394, 3.595) / 0.03)
# print("p2", chi2.pdf(0.03 * 400, 2.2870, 4.5309, 26.5874) / 0.03)
# print("x3", 2.43 * 10 ** -5 * math.exp(11.24 * 0.97) + 0.187)
# S3FIFOSimulator(0.97, chi2.pdf(0.03 * 65, 4.4912, 1.1394, 3.595) / 0.03,
#                 chi2.pdf(0.03 * 400, 2.2870, 4.5309, 26.5874) / 0.03,
#                 2.43 * 10 ** -5 * math.exp(11.24 * 0.97) + 0.187, 72, 5).run(100000)
#
# print("p1", chi2.pdf(0.02 * 65, 4.4912, 1.1394, 3.595) / 0.02)
# print("p2", chi2.pdf(0.02 * 400, 2.2870, 4.5309, 26.5874) / 0.02)
# print("x3", 2.43 * 10 ** -5 * math.exp(11.24 * 0.98) + 0.187)
# S3FIFOSimulator(0.98, chi2.pdf(0.02 * 65, 4.4912, 1.1394, 3.595) / 0.02,
#                 chi2.pdf(0.02 * 400, 2.2870, 4.5309, 26.5874) / 0.02,
#                 2.43 * 10 ** -5 * math.exp(11.24 * 0.98) + 0.187, 72, 5).run(100000)  # 45M
