from datetime import datetime
from pathlib import Path
import os

class Logs:

    """
        Class that aims to ease the logging process of error
    """

    def __init__(self):
        if 'log.txt' in os.listdir():
            self.logfile = Path('./DataStore/log.txt')
        else:
            self.logfile = Path('../DataStore/log.txt')

    def log_msg(self, msg):
        try:
            if 'log.txt' in os.listdir():
                with open(Path('log.txt'), 'a') as f:
                    f.write(msg)
            else:
                with open(Path('./DataStore/log.txt'), 'a') as f:
                    f.write(msg)
        except FileNotFoundError:
            with open(Path('../DataStore/log.txt'), 'a') as f:
                f.write(msg)

    def now_date(self):
        return datetime.today()

if __name__ == '__main__':
    log_obj = Logs()
    print(log_obj.now_date())
