import logging
from datetime import datetime
from src.data.scrapers import FeatureDataCollector
from src.data.news_collector import NewsCollector
from src.features.feature_selection import FeatureSelector
from src.models.predictor import DollarPredictor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    try:
        # Inicializa os componentes
        collector = FeatureDataCollector()
        news_collector = NewsCollector()
        feature_selector = FeatureSelector()
        predictor = DollarPredictor()

        # Coleta dados históricos
        logger.info("Coletando dados históricos...")
        historical_data = collector.collect_data(days_back=365)
        
        if historical_data is None:
            raise ValueError("Falha ao coletar dados históricos")
        
        logger.info("Coletando notícias relevantes...")
        news_data = news_collector.collect_news()

        # Seleciona features
        logger.info("Selecionando features relevantes...")
        selected_features = feature_selector.select_features(historical_data, news_data)

        # Realiza predição
        logger.info("Realizando predição...")
        prediction = predictor.predict(selected_features)

        logger.info(f"Previsão do dólar para {datetime.now().date()}: R$ {prediction:.4f}")

    except Exception as e:
        logger.error(f"Erro durante a execução: {str(e)}")
        raise

if __name__ == "__main__":
    main() 