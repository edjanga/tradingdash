from datetime import datetime

class Logs:

    """
        Class that aims to ease the logging process of error
    """

    def __init__(self):
        pass

    def log_msg(self, msg):
        with open('log.txt', 'a') as f:
            f.write(msg)

    def now_date(self):
        return datetime.today()

if __name__ == '__main__':
    log_obj = Logs()
    print(log_obj.now_date())
