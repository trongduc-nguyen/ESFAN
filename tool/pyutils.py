import time
from multiprocessing.pool import ThreadPool

class AverageMeter:
    def __init__(self, *keys):
        self.__data = dict()
        for k in keys:
            self.__data[k] = [0.0, 0]

    def add(self, dict):
        for k, v in dict.items():
            if k not in self.__data:
                self.__data[k] = [0.0, 0]
            self.__data[k][0] += v
            self.__data[k][1] += 1

    def get(self, *keys):
        if len(keys) == 1:
            k = keys[0]
            if self.__data[k][1] == 0: return 0.0 # Fix lỗi chia 0
            return self.__data[k][0] / self.__data[k][1]
        else:
            v_list = []
            for k in keys:
                if self.__data[k][1] == 0:
                    v_list.append(0.0) # Fix lỗi chia 0
                else:
                    v_list.append(self.__data[k][0] / self.__data[k][1])
            return tuple(v_list)

    def pop(self, key=None):
        if key is None:
            v = self.get(*self.__data.keys())
            self.reset()
            return v
        else:
            v = self.get(key)
            self.__data[key] = [0.0, 0]
            return v

    # --- BỔ SUNG HÀM RESET ---
    def reset(self):
        for k in self.__data.keys():
            self.__data[k] = [0.0, 0]


class Timer:
    def __init__(self, starting_msg = None):
        self.start = time.time()
        self.stage_start = self.start

        if starting_msg is not None:
            print(starting_msg, time.ctime(time.time()))

    def update_progress(self, progress):
        self.elapsed = time.time() - self.start
        if progress > 0:
            self.est_total = self.elapsed / progress
        else:
            self.est_total = 0
        self.est_remaining = self.est_total - self.elapsed
        self.est_finish = int(self.start + self.est_total)

    def str_est_finish(self):
        return str(time.ctime(self.est_finish))

    def get_stage_elapsed(self):
        return time.time() - self.stage_start

    def reset_stage(self):
        self.stage_start = time.time()


class BatchThreader:
    def __init__(self, func, args_list, batch_size, prefetch_size=4, processes=1):
        self.batch_size = batch_size
        self.prefetch_size = prefetch_size

        self.pool = ThreadPool(processes=processes)
        self.async_result = []

        self.func = func
        self.left_args_list = args_list
        self.n_tasks = len(args_list)

        # initial work
        self.__start_works(self.__get_n_pending_works())

    def __start_works(self, times):
        for _ in range(times):
            args = self.left_args_list.pop(0)
            self.async_result.append(
                self.pool.apply_async(self.func, args))

    def __get_n_pending_works(self):
        return min((self.prefetch_size + 1) * self.batch_size - len(self.async_result)
                   , len(self.left_args_list))

    def pop_results(self):
        n_inwork = len(self.async_result)

        n_fetch = min(n_inwork, self.batch_size)
        rtn = [self.async_result.pop(0).get()
                for _ in range(n_fetch)]

        to_fill = self.__get_n_pending_works()
        if to_fill == 0:
            self.pool.close()
        else:
            self.__start_works(to_fill)

        return rtn