from logging import Logger, StreamHandler, Formatter, DEBUG

logger = Logger("tilelang-ascend")
handler = StreamHandler()
handler.setLevel(DEBUG)
formatter = Formatter("[%(asctime)s] [%(levelname)s] %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(DEBUG)


def log_debug(msg: str):
    logger.debug(msg)

def log2file(file:str, msg: str):
    with open(file, 'w') as f:
        f.write(msg + '\n')
        