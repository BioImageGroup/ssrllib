import datetime


def print_ts(text):
    """
    Prints text to stdout and includes timestamps at the beginning of each line
    """
    print('[%s] %s' % (datetime.datetime.now(), text), flush=True)
