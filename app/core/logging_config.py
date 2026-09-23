import logging

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: str = "INFO") -> None:
    """Configure le logging racine de l'application. À appeler une fois au démarrage."""
    logging.basicConfig(level=level, format=LOG_FORMAT, datefmt=DATE_FORMAT, force=True)

    # httpx (utilisé par le SDK Gemini) logge chaque requête HTTP : on le bride pour
    # garder des logs lisibles, même en DEBUG. Le logger "google_genai" reste actif
    # car il signale les tentatives de retry, utiles pour diagnostiquer la latence.
    for noisy_logger in ("httpx", "httpcore"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)
