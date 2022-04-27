from collections import deque

class History:
    def __init__(self, max_history_saved):
        self._max_history_saved = max_history_saved
        self.__history = deque()

    def __iter__(self):
        return self.__history.__iter__()

    def append(self, __object) -> None:
        self.__history.append(__object)

        if len(self.__history) > self._max_history_saved >= 0:
            self.__history.popleft()

