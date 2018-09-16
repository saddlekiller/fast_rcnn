import logging
import logging.handlers
import os
import sys
import warnings

# FORMAT = '%(asctime)s [%(filename)s:%(lineno)s] %(message)s'
FORMAT = '%(message)s'
SHOW_LENGTH = 150
RATE = 0.4
LEFT = str(int(RATE * SHOW_LENGTH))
RIGHT = str(SHOW_LENGTH - int(LEFT) - 1)


class Logger(logging.Logger):

    def __init__(self, name, level=logging.NOTSET):
        super(Logger, self).__init__(name=name, level=level)
        self.maximum_show = str(SHOW_LENGTH)
        self.enable_color = True
        self.head = '|'
        self.hline_head = '+'
        self.hline_content = '-'
        self.DEBUG_FLAG = '[{:^5}] '.format('DEBUG')
        self.INFO_FLAG = ''
        self.WARN_FLAG = ''
        self.ERROR_FLAG = '[{:^5}] '.format('ERROR')
        self.FATAL_FLAG = ''

    def set_color(self, enable_color):
        self.enable_color = enable_color

    def debug(self, msg, *args, **kwargs):
        if self.isEnabledFor(logging.DEBUG):
            msg = (self.DEBUG_FLAG + self.head + ' {:^' + self.maximum_show + '} ' + self.head).format(str(msg))
            if self.enable_color:
                msg = '\033[1;32;40m{}\033[0m'.format(msg)
            self._log(logging.DEBUG, msg, args, **kwargs)

    def info(self, msg, *args, **kwargs):
        if self.isEnabledFor(logging.INFO):
            msg = (self.INFO_FLAG + self.head + ' {:^' + self.maximum_show + '} ' + self.head).format(str(msg))
            if self.enable_color:
                msg = '\033[1;34;40m{}\033[0m'.format(msg)
            self._log(logging.INFO, msg, args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        if self.isEnabledFor(logging.WARNING):
            msg = (self.WARN_FLAG + self.head + ' {:^' + self.maximum_show + '} ' + self.head).format(str(msg))
            if self.enable_color:
                msg = '\033[1;33;40m{}\033[0m'.format(msg)
            self._log(logging.WARNING, msg, args, **kwargs)

    def warn(self, msg, *args, **kwargs):
        warnings.warn("The 'warn' method is deprecated, "
                      "use 'warning' instead", DeprecationWarning, 2)
        self.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        if self.isEnabledFor(logging.ERROR):
            msg = (self.ERROR_FLAG + self.head + ' {:^' + self.maximum_show + '} ' + self.head).format(str(msg))
            if self.enable_color:
                msg = '\033[1;31;43m{}\033[0m'.format(msg)
            self._log(logging.ERROR, msg, args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        if self.isEnabledFor(logging.CRITICAL):
            msg = (self.FATAL_FLAG + self.head + ' {:^' + self.maximum_show + '} ' + self.head).format(str(msg))
            if self.enable_color:
                msg = '\033[1;31;40m{}\033[0m'.format(msg)
            self._log(logging.CRITICAL, msg, args, **kwargs)

    def hline(self, *args, **kwargs):
        if self.isEnabledFor(logging.INFO):
            msg_format = self.INFO_FLAG + self.hline_head + '{:^' + self.maximum_show + '}' + self.hline_head
            msg = (msg_format).format(self.hline_content * (int(self.maximum_show) + 2))
            if self.enable_color:
                msg = '\033[1;34;40m{}\033[0m'.format(msg)
            self._log(logging.INFO, msg, args, **kwargs)


logging.setLoggerClass(Logger)

stream_handle = logging.StreamHandler()
stream_handle.setFormatter(logging.Formatter(FORMAT))

logger = logging.getLogger('logger')
logger.set_color(True)
logger.addHandler(stream_handle)
logger.setLevel(logging.INFO)
#
# debug_logger = logging.getLogger('debug')
# debug_logger.set_color(True)
# debug_logger.addHandler(stream_handle)
# debug_logger.setLevel(logging.DEBUG)
#
# test_logger = logging.getLogger('test')
# test_logger.set_color(True)
# test_logger.addHandler(stream_handle)
# test_logger.setLevel(logging.DEBUG)
