"""
Module that sets up rapthor logging
"""
import logging


def add_coloring_to_emit_ansi(fn):
    """
    Colorize the logging output
    """
    def new(*args):
        levelno = args[0].levelno
        if (levelno >= 50):
            color = '\x1b[31m'  # red
        elif (levelno >= 40):
            color = '\x1b[31m'  # red
        elif (levelno >= 30):
            color = '\x1b[33m'  # yellow
        elif (levelno >= 20):
            color = '\x1b[32m'  # green
        elif (levelno >= 10):
            color = '\x1b[35m'  # pink
        else:
            color = '\x1b[0m'   # normal
        if isinstance(args[0].msg, Exception):
            # For exceptions, add the class name to the logged message.
            args[0].msg = "{0}: {1}".format(args[0].msg.__class__.__name__,
                                            args[0].msg)
        args[0].msg = color + args[0].msg + '\x1b[0m'
        return fn(*args)
    return new


def set_level(level):
    """
    Change verbosity of console output
    """
    if level == 'warning':
        level = logging.WARNING
    elif level == 'info':
        level = logging.INFO
    elif level == 'debug':
        level = logging.DEBUG
    else:
        level = logging.NOTSET
    logging.root.setLevel(logging.DEBUG)
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        logger.setLevel(level)
    for handler in logging.root.handlers:
        if handler.name == 'console':
            handler.setLevel(level)


class Whitelist(logging.Filter):
    """
    Filter out any non-rapthor loggers
    """
    def filter(self, record):
        if 'rapthor' in record.name and 'executable_' not in record.name:
            return True
        else:
            return False


def set_log_file(log_file):
    """
    Define and add a file handler
    """
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)  # file always logs everything
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    fh.setFormatter(formatter)
    fh.emit = add_coloring_to_emit_ansi(fh.emit)
    fh.addFilter(Whitelist())
    logging.root.addHandler(fh)


# Define and add console handler (in color)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)  # default log level
formatter = logging.Formatter('%(levelname)s - %(name)s - %(message)s')
ch.setFormatter(formatter)
ch.emit = add_coloring_to_emit_ansi(ch.emit)
ch.set_name('console')
ch.addFilter(Whitelist())
logging.root.addHandler(ch)

# Set root level (the handlers will set their own levels)
logging.root.setLevel(logging.DEBUG)
