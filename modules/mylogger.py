import os
import sys
import logging

def init_logging(logger_name: str, log_dir:str, filename = "info.log", reset=False) -> logging.Logger:
    '''
    logging levels:
        DEBUG
        INFO
        WARNING
        ERROR
        CRITICAL
    
    output INFO, WARNING, ERROR, CRITICAL to console
    output ALL to file
    '''
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    file_path = os.path.join(log_dir, filename)
    if reset and os.path.exists(file_path):
        os.remove(file_path)

    if logger.handlers == []:

        os.makedirs(log_dir, exist_ok=True)

        formatter = logging.Formatter(
            "%(asctime)s/%(levelname)s/%(name)s/%(funcName)s():%(lineno)s\n%(message)s \n"
        )

        # file handler
        fh = logging.FileHandler(str(log_dir) + "/" + filename)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)

        # console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)

        logger.propagate = False

    return logger

if __name__ == "__main__":
    logger = init_logging(__name__, log_dir='logs', filename='test_logger.log', reset=True)
    logger.debug("debug")
    logger.info("info")
    logger.warning("warning")
    logger.error("error")
    logger.critical("critical")