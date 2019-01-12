import multiprocessing

class ProcessQueue:

    def __init__(self, max_processes=6):
        self.__processes = []
        self.__max_processes = max_processes

    def schedule_process(self, func, args):
        if len(self.__processes) >= self.__max_processes:
            first_process = self.__processes.pop(0)
            first_process.join()
        queue = multiprocessing.queues.Queue()
        new_process = multiprocessing.Process(target=func, args=args)
        self.__processes.append(new_process)
        new_process.start()

    def await_all_processes(self):
        for process in self.__processes:
            process.join()


    def __run_process(func, args, queue):
        result = func(*args)
        queue.put(result)
        

