from absl import logging


def set_up_logging(log_path=None, level=logging.INFO, formatter='%(asctime)s - %(name)s - %(levelname)s - %(message)s'):
    logging.set_verbosity(level)
    # create file handler which logs even debug messages
    if log_path:
        try:
            logging.get_absl_handler().use_absl_log_file(program_name="tfnlp", log_dir=log_path)
            logging.info('Saving logs to "%s"' % log_path)
        except FileNotFoundError:
            logging.info('Cannot save logs to file in Cloud ML Engine')
