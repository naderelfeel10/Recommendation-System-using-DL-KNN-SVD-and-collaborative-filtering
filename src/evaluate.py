import sys
import os
import logging


LOG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../logs'))
os.makedirs(LOG_PATH, exist_ok=True)
LOG_FILE = os.path.join(LOG_PATH, 'evaluation.log')


logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)



logger = logging.getLogger(__name__)



sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import models.user_based_cf_model as cf_based
import models.DL_model as MF_model




def evaluate_Cf():
    try:
        logger.info("Starting user-based collaborative filtering evaluation...")

        user_based_model = cf_based.UserBasedCF()
        logger.info("Model initialized successfully.")


        precisions = user_based_model.mean_precision_at_k()
        mean_precision = sum(precisions) / len(precisions)
        logger.info(f"Mean test precision: {mean_precision:.4f}")

        print("Evaluation completed successfully. Check logs for details.")
    except Exception as e:

        logger.exception(f"Error during evaluation: {e}")
        print("An error occurred during evaluation. Check logs for details.")




def evaluate_MF():
    try:
        logger.info("Starting matrix-factorization evaluation...")

        mf = MF_model.DL_MatrixFactorization(100)
        mf.fit()
        logger.info("Model initialized successfully.")

        print("Evaluation completed successfully. Check logs for details.")
    except Exception as e:

        logger.exception(f"Error during evaluation: {e}")
        print("An error occurred during evaluation. Check logs for details.")



if __name__ == "__main__":
    evaluate_Cf()
    evaluate_MF()