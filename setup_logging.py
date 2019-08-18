import logging


def setup_logging(level: int = logging.INFO, show_timestamps: bool = False,
                  show_location: bool = True):
    logging.basicConfig(
        format='{0}%(levelname)-8s{1}%(message)s'.format(
            ('%(asctime)s ' if show_timestamps else ''),
            ('[%(filename)s:%(funcName)s:%(lineno)d] ' if show_location else '')
        ),
        datefmt='%Y-%m-%d %H:%M:%S',
        level=level,
    )
