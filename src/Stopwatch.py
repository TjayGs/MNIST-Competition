import time


class Stopwatch:
    start_stop_dict = {}

    def start(self, domain: str):
        """
        Method will set the start time of the given domain
        """
        self.start_stop_dict[domain] = {'start': time.perf_counter()}

    def stop(self, domain: str):
        """
        Method will set the time for the stop part of the domain and will print the duration
        """
        self.start_stop_dict[domain]['stop'] = time.perf_counter()
        print("Domain {} needed {}s".format(domain,
                                             self.start_stop_dict[domain]['stop'] - self.start_stop_dict[domain][
                                                 'start']))
