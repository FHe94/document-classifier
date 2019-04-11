import subprocess
import memory_profiler

class MemoryProfiler:

    def profile(self, args, interval = 0.01):
        with subprocess.Popen(args, shell=True) as process:
            memory_usage = memory_profiler.memory_usage(process, interval=interval, include_children = True)
            peak_memory_usage = max(memory_usage)
            return MemoryUsageInfo(memory_usage, peak_memory_usage, interval)

class MemoryUsageInfo:

    def __init__(self, memory_usage, peak_memory_usage, interval):
        self.memory_usage = memory_usage
        self.peak_memory_usage = peak_memory_usage
        self.interval = interval

    def __format_data_point(self, data_point):
        if data_point > 1024:
            value = data_point/1024
            unit = "Gb"
        else:
            value = data_point
            unit = "Mb"
        return "{} {}".format(round(value, 3), unit)

    def __str__(self):
        str_parts = [ "Peak usage: {}".format(self.__format_data_point(self.peak_memory_usage)) ]
        for data_point in self.memory_usage:
            str_parts.append(self.__format_data_point(data_point))
        return "\n".join(str_parts)

