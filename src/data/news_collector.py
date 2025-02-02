import logging
import os
from datetime import datetime, timedelta
import pandas as pd
import json
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod
from tqdm import tqdm
import hashlib

# Configuração de logging
logger = logging.getLogger('news_collector')

class NewsProvider(ABC):
    @abstractmethod
    def get_news(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Coleta notícias para o período especificado
        Args:
            start_date: Data inicial
            end_date: Data final
        Returns:
            DataFrame com as notícias coletadas
        """
        pass

    def _validate_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Valida e corrige datas"""
        if 'date' not in df.columns:
            return pd.DataFrame(columns=['date', 'title', 'content', 'source', 'url', 'news_id'])
            
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        return df

    def _generate_news_id(self, row: pd.Series) -> str:
        """
        Gera um ID único para cada notícia baseado no conteúdo
        Args:
            row: Série com os dados da notícia
        Returns:
            str: ID único da notícia
        """
        # Combina título e conteúdo para gerar um hash único
        content = f"{row['title']}_{row['content']}_{row['source']}_{row['date'].strftime('%Y-%m-%d')}"
        return hashlib.md5(content.encode()).hexdigest()

    def _ensure_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Garante que todas as colunas necessárias existem
        """
        if df.empty:
            return pd.DataFrame(columns=['date', 'title', 'content', 'source', 'url', 'news_id'])
            
        required_columns = ['date', 'title', 'content', 'source', 'url']
        
        # Adiciona colunas faltantes
        for col in required_columns:
            if col not in df.columns:
                if col == 'url':
                    df[col] = ''
                else:
                    df[col] = None
        
        # Gera IDs únicos para as notícias
        df['news_id'] = df.apply(self._generate_news_id, axis=1)
        
        # Seleciona apenas as colunas necessárias na ordem correta
        return df[required_columns + ['news_id']]

class NewsDataCollector:
    def __init__(self, base_path: str = ''):
        """
        Inicializa o coletor de notícias
        Args:
            base_path: Caminho base para os diretórios de dados
        """
        # Define caminhos base
        self.base_path = base_path or os.getcwd()
        self.data_dir = os.path.join(self.base_path, 'data')
        self.news_dir = os.path.join(self.data_dir, 'news')
        self.log_dir = os.path.join(self.data_dir, 'logs')
        
        # Cria diretórios
        self._ensure_directories()
        
        # Configura logging
        self._setup_logging()
        
        # Inicializa providers (você deve implementar suas próprias fontes)
        self.providers = self._initialize_providers()
        
        logger.info("NewsDataCollector inicializado com sucesso")

    def _ensure_directories(self):
        """Cria estrutura de diretórios necessária"""
        for directory in [self.data_dir, self.news_dir, self.log_dir]:
            os.makedirs(directory, exist_ok=True)
            logger.debug(f"Diretório criado/verificado: {directory}")

    def _setup_logging(self):
        """Configura o sistema de logging"""
        # Define formato do log
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        formatter = logging.Formatter(log_format)
        
        # Configura arquivo de log
        log_file = os.path.join(self.log_dir, 'news_collector.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        
        # Configura output para console
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        # Configura logger
        logger.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    def _initialize_providers(self) -> Dict[str, NewsProvider]:
        """
        Inicializa os provedores de notícias
        Returns:
            Dict com os provedores disponíveis
        """
        # Implemente seus provedores aqui
        return {}

    def _get_last_news_date(self) -> Optional[datetime]:
        """
        Obtém a data da notícia mais recente no histórico
        Returns:
            datetime ou None se não houver histórico
        """
        try:
            historical_file = os.path.join(self.news_dir, 'historical_news.csv')
            if not os.path.exists(historical_file):
                return None
            
            df = pd.read_csv(historical_file)
            if df.empty:
                return None
                
            df['date'] = pd.to_datetime(df['date'])
            return df['date'].max()
            
        except Exception as e:
            logger.error(f"Erro ao obter última data de notícias: {str(e)}")
            return None

    def collect_news(self, days_back: int = 30, end_date: Optional[datetime] = None) -> bool:
        """
        Coleta notícias de forma incremental
        Args:
            days_back: Número de dias para coletar caso não haja dados anteriores
            end_date: Data final para coleta. Se None, usa a data atual
        Returns:
            bool: True se a coleta foi bem sucedida
        """
        try:
            if not self.providers:
                logger.error("Nenhum provedor de notícias configurado")
                return False

            # Define data final
            if end_date is None:
                end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            
            # Obtém última data conhecida
            last_date = self._get_last_news_date()
            
            # Define data inicial
            if last_date:
                start_date = last_date + timedelta(days=1)
                logger.info(f"Coletando notícias a partir de {start_date.date()}")
            else:
                start_date = end_date - timedelta(days=days_back)
                logger.info(f"Coletando histórico completo de {days_back} dias")
            
            # Carrega notícias existentes
            historical_file = os.path.join(self.news_dir, 'historical_news.csv')
            existing_news = None
            if os.path.exists(historical_file):
                existing_news = pd.read_csv(historical_file)
                existing_news['date'] = pd.to_datetime(existing_news['date'])
            
            all_news = []
            # Barra de progresso
            pbar = tqdm(total=len(self.providers), desc="Coletando notícias")
            
            for source, provider in self.providers.items():
                try:
                    logger.info(f"Coletando notícias de {source}")
                    
                    # Coleta novas notícias
                    df = provider.get_news(start_date, end_date)
                    
                    if not isinstance(df, pd.DataFrame) or df.empty:
                        logger.warning(f"Nenhuma notícia encontrada para {source}")
                        continue
                    
                    # Normaliza e valida os dados
                    df = provider._validate_dates(df)
                    df = provider._ensure_columns(df)
                    
                    if not df.empty:
                        # Combina com notícias existentes
                        if existing_news is not None:
                            # Remove duplicatas baseado no ID da notícia
                            existing_ids = set(existing_news['news_id'])
                            df = df[~df['news_id'].isin(existing_ids)]
                            
                            # Adiciona novas notícias
                            if not df.empty:
                                all_news.append(df)
                                logger.info(f"Adicionadas {len(df)} novas notícias de {source}")
                        else:
                            all_news.append(df)
                            logger.info(f"Adicionadas {len(df)} notícias de {source}")
                    
                except Exception as e:
                    logger.error(f"Erro ao coletar notícias de {source}: {str(e)}")
                finally:
                    pbar.update(1)
            
            pbar.close()

            # Se não há novas notícias, mantém o arquivo existente
            if not all_news:
                logger.info("Nenhuma notícia nova encontrada")
                return True

            # Combina todas as notícias
            new_news = pd.concat(all_news, ignore_index=True)
            
            # Combina com notícias existentes
            if existing_news is not None:
                final_news = pd.concat([existing_news, new_news], ignore_index=True)
            else:
                final_news = new_news
            
            # Remove duplicatas finais
            final_news = final_news.sort_values('date')
            final_news = final_news.drop_duplicates(subset=['news_id'], keep='last')
            
            # Salva arquivo histórico
            final_news.to_csv(historical_file, index=False)
            logger.info(f"Histórico de notícias atualizado em {historical_file}")
            
            # Salva metadados
            metadata = {
                'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'total_news': len(final_news),
                'sources': final_news['source'].unique().tolist(),
                'date_range': {
                    'start': final_news['date'].min().strftime('%Y-%m-%d'),
                    'end': final_news['date'].max().strftime('%Y-%m-%d')
                },
                'news_per_source': final_news['source'].value_counts().to_dict()
            }
            
            metadata_file = os.path.join(self.news_dir, 'news_metadata.json')
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Metadados salvos em {metadata_file}")
            logger.info(f"Total de notícias no histórico: {len(final_news)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Erro na coleta de notícias: {str(e)}")
            return False 