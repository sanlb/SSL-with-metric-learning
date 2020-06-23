import logging

class GenericCSV(object):
    # This code is referenced from https://github.com/smlaine2/tempens/blob/master/report.py
    def __init__(self, fname, *fields):
        self.fields = fields
        self.fout = open(fname, 'wt')
        self.fout.write(",".join(fields) + '\n')
        self.fout.flush()

    def add_data(self, *values):
        assert len(values) == len(self.fields)
        strings = [v if isinstance(v, str) else '%g' % v for v in values]
        self.fout.write(",".join(strings) + '\n')
        self.fout.flush()

    def close(self):
        self.fout.close()

    def __enter__(self): # for 'with' statement
        return self

    def __exit__(self, *excinfo):
        self.close()

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger
