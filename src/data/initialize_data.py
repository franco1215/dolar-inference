import logging
import os
from datetime import datetime
from scrapers import FeatureDataCollector
import pandas as pd
import json
from typing import Tuple, Dict, Any

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/data_initialization.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class DataInitializer:
    def __init__(self, lookback_years: int = 5):
        """
        Inicializa o DataInitializer
        Args:
            lookback_years (int): Anos de dados históricos para coletar
        """
        self.lookback_years = lookback_years
        
        # Define caminhos base
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.data_dir = os.path.join(self.base_dir, 'data')
        
        # Define subdiretórios
        self.raw_dir = os.path.join(self.data_dir, 'raw')
        self.processed_dir = os.path.join(self.data_dir, 'processed')
        self.metadata_dir = os.path.join(self.data_dir, 'metadata')
        self.logs_dir = os.path.join(self.data_dir, 'logs')
        
        # Garante que os diretórios existem
        for directory in [self.raw_dir, self.processed_dir, self.metadata_dir, self.logs_dir]:
            os.makedirs(directory, exist_ok=True)

        # Inicializa o coletor
        self.scraper = FeatureDataCollector()

    def _ensure_directories(self):
        """Cria estrutura de diretórios necessária"""
        directories = ['data', 'data/raw', 'data/processed']
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def _validate_data(self, df: pd.DataFrame) -> bool:
        """
        Valida os dados coletados
        Args:
            df (pd.DataFrame): DataFrame com dados coletados
        Returns:
            bool: True se dados são válidos
        """
        if df.empty:
            logger.error("Dados coletados estão vazios")
            return False

        # Verifica se pelo menos temos a data e valor
        required_columns = ['Date']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Colunas essenciais faltando: {missing_columns}")
            return False

        # Verifica valores nulos
        null_counts = df.isnull().sum()
        if null_counts.any():
            logger.warning(f"Valores nulos encontrados:\n{null_counts[null_counts > 0]}")
            # Não falha por valores nulos, apenas avisa

        return True

    def _save_metadata(self, df: pd.DataFrame):
        """
        Salva metadados da coleta
        Args:
            df (pd.DataFrame): DataFrame com dados coletados
        """
        metadata = {
            'collection_date': datetime.now().isoformat(),
            'lookback_years': self.lookback_years,
            'total_records': len(df),
            'features_collected': df['feature_code'].unique().tolist(),
            'date_range': {
                'start': df['Date'].min().isoformat(),
                'end': df['Date'].max().isoformat()
            },
            'sources': df['source'].unique().tolist()
        }

        metadata_file = 'data/metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Metadados salvos em {metadata_file}")

    def _is_market_open(self) -> bool:
        """
        Verifica se o mercado de câmbio está aberto no Brasil
        Returns:
            bool: True se o mercado estiver aberto
        """
        now = datetime.now()
        
        # Verifica se é fim de semana
        if now.weekday() >= 5:  # 5 = Sábado, 6 = Domingo
            return False
            
        # Horário do mercado de câmbio (9h às 17h)
        market_start = now.replace(hour=9, minute=0, second=0, microsecond=0)
        market_end = now.replace(hour=17, minute=0, second=0, microsecond=0)
        
        return market_start <= now <= market_end

    def _get_end_date(self) -> datetime:
        """
        Define a data final para coleta considerando o horário do mercado
        Returns:
            datetime: Data final para coleta
        """
        now = datetime.now()
        
        # Se o mercado não estiver aberto, usa o dia anterior
        if not self._is_market_open():
            # Se for segunda-feira e o mercado ainda não abriu, volta para sexta
            if now.weekday() == 0 and now.hour < 9:
                days_to_subtract = 3
            # Se for outro dia e o mercado ainda não abriu
            elif now.hour < 9:
                days_to_subtract = 1
            # Se o mercado já fechou
            elif now.hour >= 17:
                days_to_subtract = 1
            # Se for fim de semana
            elif now.weekday() >= 5:
                days_to_subtract = now.weekday() - 4  # Volta para sexta-feira
            else:
                days_to_subtract = 0
                
            now = now - pd.Timedelta(days=days_to_subtract)
        
        # Normaliza para meia-noite
        return now.replace(hour=0, minute=0, second=0, microsecond=0)

    def initialize_historical_data(self) -> bool:
        """
        Carrega dados históricos iniciais
        Returns:
            bool: True se inicialização foi bem sucedida
        """
        try:
            logger.info(f"Iniciando coleta de {self.lookback_years} anos de dados históricos...")
            
            # Define data final baseada no horário do mercado
            end_date = self._get_end_date()
            logger.info(f"Data final para coleta: {end_date.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Coleta dados
            days = self.lookback_years * 365
            df = self.scraper.collect_data(days_back=days, end_date=end_date)
            
            # Verifica se dados foram coletados
            if df is None or df.empty:
                logger.error("Falha na coleta de dados ou dados vazios")
                return False
            
            # Valida dados básicos
            if not self._validate_data(df):
                logger.error("Falha na validação dos dados")
                return False
            
            # Verifica features coletadas vs. configuradas
            configured_features = [str(feature.get('code', '')) for feature in self.scraper.features_config]
            collected_features = [col for col in df.columns if col != 'Date']
            missing_features = [f for f in configured_features if f not in collected_features]
            
            if missing_features:
                logger.warning(f"Features não coletadas: {missing_features}")
                logger.warning("Continuando processamento com features disponíveis...")
            
            # Log das features coletadas com sucesso
            logger.info(f"Features coletadas com sucesso: {collected_features}")
            
            # Salva dados brutos
            timestamp = datetime.now().strftime("%Y%m%d")
            raw_file = os.path.join('data', 'raw', f'historical_data_{timestamp}.csv')
            df.to_csv(raw_file, index=False)
            logger.info(f"Dados brutos salvos em {raw_file}")
            
            try:
                # Processa e limpa dados
                df_processed, metadata = self._process_data(df)
                
                if df_processed.empty:
                    logger.error("Processamento resultou em DataFrame vazio")
                    return False
                
                # Adiciona informações sobre features aos metadados
                metadata['feature_status'] = {
                    'configured': configured_features,
                    'collected': collected_features,
                    'missing': missing_features,
                    'success_rate': f"{(len(collected_features) / len(configured_features)) * 100:.2f}%"
                }
                
                # Adiciona informação sobre o horário da última atualização
                metadata['last_update'] = {
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'market_status': 'open' if self._is_market_open() else 'closed',
                    'last_trading_date': end_date.strftime('%Y-%m-%d')
                }
                
                # Salva dados processados
                processed_file = os.path.join('data', 'processed', f'historical_data_{timestamp}.csv')
                df_processed.to_csv(processed_file, index=False)
                logger.info(f"Dados processados salvos em {processed_file}")
                
                # Salva metadados
                metadata_file = os.path.join('data', 'processed', f'metadata_{timestamp}.json')
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
                logger.info(f"Metadados salvos em {metadata_file}")
                
                # Log final com estatísticas
                logger.info(f"Processamento concluído:")
                logger.info(f"- Total de features configuradas: {len(configured_features)}")
                logger.info(f"- Features coletadas com sucesso: {len(collected_features)}")
                logger.info(f"- Features não coletadas: {len(missing_features)}")
                logger.info(f"- Taxa de sucesso: {metadata['feature_status']['success_rate']}")
                logger.info(f"- Última data de negociação: {end_date.strftime('%Y-%m-%d')}")
                
                return True
                
            except Exception as e:
                logger.error(f"Erro durante o processamento: {str(e)}")
                return False

        except Exception as e:
            logger.error(f"Erro na inicialização dos dados: {str(e)}")
            return False

    def _process_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Processa os dados brutos para o formato final, removendo dias sem negociação
        Args:
            df: DataFrame com os dados brutos
        Returns:
            Tuple[DataFrame processado, Dict com metadados]
        """
        try:
            if df.empty:
                logger.error("DataFrame vazio recebido para processamento")
                return pd.DataFrame(), {}

            # Faz uma cópia para evitar modificações no DataFrame original
            df_copy = df.copy()
            
            # Verifica se o DataFrame está no formato correto (já pivotado)
            if 'Value' in df_copy.columns and 'feature_code' in df_copy.columns:
                logger.info("Convertendo DataFrame do formato longo para largo")
                # Converte datas para datetime sem timezone
                df_copy['Date'] = pd.to_datetime(df_copy['Date']).dt.tz_localize(None)
                
                # Garante tipos corretos antes do pivot
                df_copy['Value'] = pd.to_numeric(df_copy['Value'], errors='coerce')
                df_copy['feature_code'] = df_copy['feature_code'].astype(str)
                
                # Remove valores nulos e duplicatas
                df_copy = df_copy.dropna(subset=['Date', 'Value'])
                df_copy = df_copy.sort_values(['Date', 'feature_code'])
                df_copy = df_copy.drop_duplicates(subset=['Date', 'feature_code'], keep='last')
                
                # Cria DataFrame pivotado
                try:
                    df_copy = df_copy.pivot(
                        index='Date',
                        columns='feature_code',
                        values='Value'
                    )
                except ValueError as e:
                    logger.error(f"Erro ao criar pivot table: {str(e)}")
                    return pd.DataFrame(), {}
            else:
                logger.info("DataFrame já está no formato largo")
                # Garante que a coluna Date é o índice
                if 'Date' in df_copy.columns:
                    df_copy.set_index('Date', inplace=True)
            
            # Identifica dias úteis reais (com dados)
            valid_dates = df_copy.index[df_copy.notna().any(axis=1)]
            
            # Cria metadados antes da interpolação
            metadata = {
                'start_date': valid_dates.min().strftime('%Y-%m-%d'),
                'end_date': valid_dates.max().strftime('%Y-%m-%d'),
                'features': {},
                'total_samples': len(valid_dates),
                'trading_days_info': {
                    'total_calendar_days': (valid_dates.max() - valid_dates.min()).days + 1,
                    'total_trading_days': len(valid_dates),
                    'trading_days_percentage': f"{(len(valid_dates) / ((valid_dates.max() - valid_dates.min()).days + 1)) * 100:.2f}%"
                },
                'missing_percentages': {}
            }
            
            # Reindexar apenas com as datas válidas
            df_copy = df_copy.loc[valid_dates]
            
            # Processa cada feature individualmente
            for col in df_copy.columns:
                # Calcula estatísticas antes da interpolação
                total_values = len(df_copy[col])
                missing_values = df_copy[col].isna().sum()
                missing_pct = (missing_values / total_values) * 100
                
                # Interpolação com limite de 5 dias úteis
                df_copy[col] = (
                    df_copy[col]
                    .interpolate(method='linear', limit_direction='both', limit=5)
                )
                
                # Preenche valores restantes com a média da coluna
                col_mean = df_copy[col].mean()
                df_copy[col] = df_copy[col].fillna(col_mean)
                
                # Atualiza metadados
                metadata['features'][col] = {
                    'missing_values': int(missing_values),
                    'total_values': int(total_values),
                    'missing_percentage': float(missing_pct),
                    'mean_value': float(col_mean),
                    'std_value': float(df_copy[col].std()),
                    'min_value': float(df_copy[col].min()),
                    'max_value': float(df_copy[col].max())
                }
                metadata['missing_percentages'][col] = float(missing_pct)
            
            # Reset do índice mantendo o nome da coluna de data
            df_copy = df_copy.reset_index()
            df_copy = df_copy.rename(columns={'index': 'Date'})
            
            # Garante que a coluna Date está no formato correto
            df_copy['Date'] = pd.to_datetime(df_copy['Date']).dt.date
            
            # Ordena as colunas alfabeticamente (exceto 'Date')
            feature_cols = sorted([col for col in df_copy.columns if col != 'Date'])
            df_copy = df_copy[['Date'] + feature_cols]
            
            # Verifica a qualidade dos dados processados
            if df_copy.empty:
                logger.error("DataFrame processado está vazio")
                return pd.DataFrame(), {}
                
            missing_total = df_copy.iloc[:, 1:].isna().sum().sum()
            if missing_total > 0:
                logger.warning(f"Ainda existem {missing_total} valores faltantes após o processamento")
            
            # Adiciona informações sobre a distribuição temporal dos dados
            metadata['temporal_distribution'] = {
                'days_between_samples': {
                    'mean': f"{df_copy['Date'].diff().dt.days.mean():.2f}",
                    'max': int(df_copy['Date'].diff().dt.days.max()),
                    'min': int(df_copy['Date'].diff().dt.days.min())
                }
            }
            
            return df_copy, metadata
            
        except Exception as e:
            logger.error(f"Erro no processamento dos dados: {str(e)}")
            return pd.DataFrame(), {}

if __name__ == "__main__":
    # Inicializa com 5 anos de dados históricos
    initializer = DataInitializer(lookback_years=5)
    success = initializer.initialize_historical_data()
    
    if success:
        logger.info("Inicialização dos dados históricos concluída com sucesso!")
    else:
        logger.error("Falha na inicialização dos dados históricos") 