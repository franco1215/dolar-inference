import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import logging
import os
from abc import ABC, abstractmethod
import yfinance as yf
from bcb import sgs
from tqdm import tqdm
import requests
import time
from pandas_datareader import data as pdr

# Configuração inicial do logger
logger = logging.getLogger('feature_collector')

class DataProvider(ABC):
    @abstractmethod
    def get_data(self, feature: Dict[str, Any], start_date: datetime, end_date: datetime) -> pd.DataFrame:
        pass

    def _validate_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Valida e corrige datas"""
        if 'Date' not in df.columns:
            return pd.DataFrame(columns=['Date', 'Value', 'feature_code', 'source', 'description'])
            
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        df = df.drop_duplicates('Date', keep='last')
        return df

    def _validate_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Valida e corrige valores"""
        if 'Value' not in df.columns:
            return pd.DataFrame(columns=['Date', 'Value', 'feature_code', 'source', 'description'])
            
        df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
        df = df.dropna(subset=['Value'])
        return df
        
    def _ensure_columns(self, df: pd.DataFrame, feature: Dict[str, Any]) -> pd.DataFrame:
        """
        Garante que todas as colunas necessárias existem e estão preenchidas
        """
        if df.empty:
            return pd.DataFrame(columns=['Date', 'Value', 'feature_code', 'source', 'description'])
            
        required_columns = ['Date', 'Value', 'feature_code', 'source', 'description']
        
        # Adiciona colunas faltantes
        for col in required_columns:
            if col not in df.columns:
                if col == 'feature_code':
                    df[col] = feature.get('code', '')
                elif col == 'source':
                    df[col] = feature.get('source', '')
                elif col == 'description':
                    df[col] = feature.get('description', '')
                elif col == 'Value':
                    df[col] = 0.0
                else:  # Date
                    df[col] = pd.NaT
        
        # Garante que não há valores nulos nas colunas obrigatórias
        df['feature_code'] = df['feature_code'].fillna('')
        df['source'] = df['source'].fillna('')
        df['description'] = df['description'].fillna('')
        
        # Seleciona apenas as colunas necessárias na ordem correta
        return df[required_columns]

class BCBProvider(DataProvider):
    def get_data(self, feature: Dict[str, Any], start_date: datetime, end_date: datetime) -> pd.DataFrame:
        try:
            logger.info(f"Coletando {feature['description']} do BCB")
            series_id = feature.get('series_id')
            if not series_id:
                logger.error(f"series_id não encontrado para {feature['code']}")
                return pd.DataFrame(columns=['Date', 'Value', 'feature_code', 'source', 'description'])
            
            # Tenta várias vezes em caso de erro de conexão
            for attempt in range(3):
                try:
                    df = sgs.get({int(series_id): int(series_id)}, start=start_date, end=end_date)
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        df = df.reset_index()
                        df.columns = ['Date', 'Value']
                        df = self._validate_dates(df)
                        df = self._validate_values(df)
                        df = self._ensure_columns(df, feature)
                        return df
                except Exception as e:
                    if attempt == 2:  # Última tentativa
                        raise e
                    logger.warning(f"Tentativa {attempt + 1} falhou, tentando novamente...")
                    continue
            
            logger.warning(f"Nenhum dado encontrado no BCB para {feature['code']}")
            return pd.DataFrame(columns=['Date', 'Value', 'feature_code', 'source', 'description'])
            
        except Exception as e:
            logger.error(f"Erro BCB para {feature['code']}: {str(e)}")
            return pd.DataFrame(columns=['Date', 'Value', 'feature_code', 'source', 'description'])

    def _handle_bcb_error(self, feature: Dict[str, Any], start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Trata erros específicos do BCB e tenta métodos alternativos de coleta
        """
        series_id = feature.get('series_id')
        alt_id = feature.get('alternative_id')
        code = feature.get('code')
        
        try:
            # Primeiro tenta com o ID alternativo se disponível
            if alt_id:
                try:
                    df = sgs.get_serie(alt_id, start=start_date, end=end_date)
                    if not df.empty:
                        df = pd.DataFrame({
                            'Date': df.index,
                            'Value': df.values,
                            'feature_code': code,
                            'source': 'BCB',
                            'description': feature.get('description', '')
                        })
                        return self._validate_dates(df)
                except Exception as e:
                    logger.warning(f"Tentativa com ID alternativo falhou para {code}: {str(e)}")

            # Tenta com período reduzido
            reduced_start = end_date - timedelta(days=365)  # Reduz para 1 ano
            try:
                df = sgs.get_serie(series_id, start=reduced_start, end=end_date)
                if not df.empty:
                    df = pd.DataFrame({
                        'Date': df.index,
                        'Value': df.values,
                        'feature_code': code,
                        'source': 'BCB',
                        'description': feature.get('description', '')
                    })
                    return self._validate_dates(df)
            except Exception as e:
                logger.warning(f"Tentativa com período reduzido falhou para {code}: {str(e)}")

            # Tenta com método alternativo de coleta
            try:
                url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{series_id}/dados"
                params = {
                    'formato': 'json',
                    'dataInicial': start_date.strftime('%d/%m/%Y'),
                    'dataFinal': end_date.strftime('%d/%m/%Y')
                }
                
                response = requests.get(url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    df = pd.DataFrame(data)
                    if not df.empty:
                        df = df.rename(columns={'data': 'Date', 'valor': 'Value'})
                        df['feature_code'] = code
                        df['source'] = 'BCB'
                        df['description'] = feature.get('description', '')
                        return self._validate_dates(df)
            except Exception as e:
                logger.warning(f"Método alternativo de coleta falhou para {code}: {str(e)}")

            return pd.DataFrame()

        except Exception as e:
            logger.error(f"Erro no tratamento da feature {code}: {str(e)}")
            return pd.DataFrame()

class YahooProvider(DataProvider):
    def get_data(self, feature: Dict[str, Any], start_date: datetime, end_date: datetime) -> pd.DataFrame:
        try:
            logger.info(f"Coletando {feature['description']} do Yahoo")
            
            # Tenta várias vezes em caso de erro de conexão
            for attempt in range(3):
                try:
                    ticker = yf.Ticker(feature['symbol'])
                    df = ticker.history(
                        start=start_date,
                        end=end_date,
                        interval='1d',
                        auto_adjust=True
                    )
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        df = df.reset_index()
                        df = df[['Date', 'Close']].rename(columns={'Close': 'Value'})
                        df = self._validate_dates(df)
                        df = self._validate_values(df)
                        df = self._ensure_columns(df, feature)
                        return df
                except Exception as e:
                    if attempt == 2:  # Última tentativa
                        raise e
                    logger.warning(f"Tentativa {attempt + 1} falhou, tentando novamente...")
                    continue
            
            logger.warning(f"Nenhum dado encontrado no Yahoo para {feature['code']}")
            return pd.DataFrame(columns=['Date', 'Value', 'feature_code', 'source', 'description'])
            
        except Exception as e:
            logger.error(f"Erro Yahoo para {feature['code']}: {str(e)}")
            return pd.DataFrame(columns=['Date', 'Value', 'feature_code', 'source', 'description'])

    def _normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            if df.empty:
                return df

            # Remove timezone das datas
            df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
            
            # Normaliza valores
            if 'Value' in df.columns:
                # Remove outliers usando z-score
                z_scores = np.abs((df['Value'] - df['Value'].mean()) / df['Value'].std())
                df = df[z_scores < 3].copy()  # Usa .copy() para evitar SettingWithCopyWarning
                
                # Preenche valores faltantes
                df.loc[:, 'Value'] = df.groupby('feature_code')['Value'].transform(
                    lambda x: x.fillna(method='ffill').fillna(method='bfill')
                )

            return df
        except Exception as e:
            logger.error(f"Erro na normalização: {str(e)}")
            return df

    def _handle_yahoo_error(self, feature: Dict[str, Any], start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Trata erros específicos do Yahoo Finance e tenta métodos alternativos
        """
        symbol = feature.get('series_id')
        code = feature.get('code')
        
        try:
            # Tenta primeiro com yfinance
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start_date, end=end_date, interval='1d')
                if not df.empty:
                    df = pd.DataFrame({
                        'Date': df.index,
                        'Value': df['Close'],
                        'feature_code': code,
                        'source': 'YAHOO',
                        'description': feature.get('description', '')
                    })
                    return self._validate_dates(df)
            except Exception as e:
                logger.warning(f"Tentativa com yfinance falhou para {code}: {str(e)}")

            # Tenta com pandas_datareader
            try:
                df = pdr.get_data_yahoo(symbol, start=start_date, end=end_date)
                if not df.empty:
                    df = pd.DataFrame({
                        'Date': df.index,
                        'Value': df['Close'],
                        'feature_code': code,
                        'source': 'YAHOO',
                        'description': feature.get('description', '')
                    })
                    return self._validate_dates(df)
            except Exception as e:
                logger.warning(f"Tentativa com pandas_datareader falhou para {code}: {str(e)}")

            return pd.DataFrame()

        except Exception as e:
            logger.error(f"Erro no tratamento da feature {code}: {str(e)}")
            return pd.DataFrame()

class FeatureDataCollector:
    def __init__(self):
        """Inicializa o coletor de features"""
        # Define o caminho base do projeto
        self.base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # Define os diretórios importantes
        self.data_dir = os.path.join(self.base_path, 'data')
        self.logs_dir = os.path.join(self.data_dir, 'logs')
        self.processed_dir = os.path.join(self.data_dir, 'processed')
        
        # Garante que os diretórios existem
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        
        # Configura logging e carrega features
        self._setup_logging()
        self.features_config = self._load_features()
        
        logger.info("FeatureDataCollector inicializado com sucesso")

    def _setup_logging(self):
        """Configura o sistema de logging"""
        global logger
        
        # Define formato do log
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        formatter = logging.Formatter(log_format)
        
        # Configura arquivo de log
        log_file = os.path.join(self.logs_dir, 'collector.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        
        # Configura output para console
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        # Configura logger
        logger = logging.getLogger('feature_collector')
        logger.setLevel(logging.INFO)
        
        # Remove handlers existentes
        logger.handlers = []
        
        # Adiciona os novos handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        logger.info("Sistema de logging configurado")

    def collect_data(self, days_back: int, end_date: datetime = None) -> pd.DataFrame:
        """
        Coleta dados para todas as features configuradas com múltiplas tentativas
        """
        try:
            all_data = []
            failed_features = []
            
            for feature in tqdm(self.features_config, desc="Coletando dados"):
                try:
                    source = feature.get('source', '').upper()
                    code = feature.get('code', '')
                    
                    # Primeira tentativa: fonte primária
                    df = self._collect_primary_source(feature, days_back, end_date)
                    if df is not None and not df.empty:
                        df = self._validate_and_process_data(df, code)
                        if not df.empty:
                            df['feature_code'] = code
                            df['source'] = source
                            all_data.append(df)
                            logger.info(f"Dados coletados com sucesso para {code}")
                            continue
                    
                    # Segunda tentativa: fonte alternativa
                    df = self._collect_alternative_source(feature, days_back, end_date)
                    if df is not None and not df.empty:
                        df = self._validate_and_process_data(df, code)
                        if not df.empty:
                            df['feature_code'] = code
                            df['source'] = source
                            all_data.append(df)
                            logger.info(f"Dados coletados com sucesso para {code} (fonte alternativa)")
                            continue
                    
                    # Se chegou aqui, falhou em todas as tentativas
                    failed_features.append(code)
                    logger.warning(f"Falha na coleta de dados para {code}")
                    
                except Exception as e:
                    failed_features.append(code)
                    logger.error(f"Erro ao coletar {code}: {str(e)}")
            
            if failed_features:
                logger.warning(f"Features não coletadas: {failed_features}")
            
            if not all_data:
                logger.error("Nenhum dado coletado com sucesso")
                return pd.DataFrame()
            
            final_df = pd.concat(all_data, ignore_index=True)
            final_df['Date'] = pd.to_datetime(final_df['Date']).dt.date
            
            return final_df
            
        except Exception as e:
            logger.error(f"Erro na coleta de dados: {str(e)}")
            return pd.DataFrame()

    def _collect_primary_source(self, feature: Dict[str, Any], days_back: int, end_date: datetime) -> Optional[pd.DataFrame]:
        """Coleta dados da fonte primária"""
        source = feature.get('source', '').upper()
        
        if source == 'BCB':
            return self._collect_bcb_data(
                series_id=feature['series_id'],
                start_date=(datetime.now() - pd.Timedelta(days=days_back)),
                end_date=end_date or datetime.now()
            )
        elif source == 'YAHOO':
            return self._collect_yahoo_data(
                symbol=feature['series_id'],
                days=days_back
            )
        return None

    def _collect_alternative_source(self, feature: Dict[str, Any], days_back: int, end_date: datetime) -> Optional[pd.DataFrame]:
        """Coleta dados de fontes alternativas quando a primária falha"""
        try:
            code = feature.get('code', '')
            
            # Configurações específicas por feature
            if code == 'DOLAR':
                return self._collect_dolar_alternative(days_back, end_date)
            elif code in ['SOJA_US', 'MILHO_US']:
                return self._collect_commodity_alternative(feature, days_back, end_date)
            elif code == 'CDS':
                return self._collect_cds_alternative(days_back, end_date)
            
            # Para features do BCB, tenta método alternativo
            if feature.get('source', '').upper() == 'BCB':
                return self._collect_bcb_alternative(feature['series_id'], days_back, end_date)
            
            return None
        except Exception as e:
            logger.error(f"Erro na coleta alternativa para {feature.get('code')}: {str(e)}")
            return None

    def _collect_bcb_data(self, series_id: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """Coleta dados do BCB com múltiplas estratégias"""
        try:
            # Primeira tentativa: usando biblioteca bcb
            try:
                series = sgs.get_serie(code=int(series_id), start=start_date, end=end_date)
                if not series.empty:
                    df = pd.DataFrame({'Date': series.index, 'Value': series.values})
                    return df
            except:
                pass

            # Segunda tentativa: API REST direta
            start_str = start_date.strftime('%d/%m/%Y')
            end_str = end_date.strftime('%d/%m/%Y')
            
            url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{series_id}/dados"
            params = {
                'formato': 'json',
                'dataInicial': start_str,
                'dataFinal': end_str
            }
            
            headers = {
                'Accept': 'application/json',
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            if data:
                df = pd.DataFrame(data)
                df.columns = ['Date', 'Value']
                df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
                df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
                return df
            
            return None
            
        except Exception as e:
            logger.error(f"Erro na coleta do BCB (série {series_id}): {str(e)}")
            return None

    def _collect_yahoo_data(self, symbol: str, days: int) -> Optional[pd.DataFrame]:
        """Coleta dados do Yahoo Finance com múltiplas estratégias"""
        try:
            end_date = datetime.now()
            start_date = end_date - pd.Timedelta(days=days)
            
            # Primeira tentativa: yfinance
            try:
                yf.pdr_override()
                df = pdr.get_data_yahoo(symbol, start=start_date, end=end_date)
                if not df.empty:
                    df = df[['Close']].reset_index()
                    df.columns = ['Date', 'Value']
                    return df
            except:
                pass

            # Segunda tentativa: yfinance direta
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval='1d')
            
            if not df.empty:
                df = df[['Close']].reset_index()
                df.columns = ['Date', 'Value']
                return df
            
            return None
            
        except Exception as e:
            logger.error(f"Erro na coleta do Yahoo (símbolo {symbol}): {str(e)}")
            return None

    def _collect_dolar_alternative(self, days_back: int, end_date: datetime) -> Optional[pd.DataFrame]:
        """Coleta dados do dólar de fontes alternativas"""
        try:
            # Tenta BCB primeiro
            series_id = "1" # Código da série do dólar no BCB
            return self._collect_bcb_data(series_id, 
                                        datetime.now() - pd.Timedelta(days=days_back),
                                        end_date or datetime.now())
        except:
            return None

    def _collect_commodity_alternative(self, feature: Dict[str, Any], days_back: int, end_date: datetime) -> Optional[pd.DataFrame]:
        """Coleta dados de commodities de fontes alternativas"""
        try:
            # Mapeia códigos alternativos
            alternative_symbols = {
                'SOJA_US': 'GC=F',  # Símbolo alternativo para soja
                'MILHO_US': 'C=F'   # Símbolo alternativo para milho
            }
            
            if feature['code'] in alternative_symbols:
                return self._collect_yahoo_data(
                    alternative_symbols[feature['code']], 
                    days_back
                )
            return None
        except:
            return None

    def _collect_cds_alternative(self, days_back: int, end_date: datetime) -> Optional[pd.DataFrame]:
        """Coleta dados de CDS de fontes alternativas"""
        try:
            # Implemente a coleta de dados de CDS de fontes alternativas
            # Esta é uma implementação vazia, pois a coleta de CDS não é suportada pela biblioteca bcb
            return None
        except Exception as e:
            logger.error(f"Erro na coleta alternativa de CDS: {str(e)}")
            return None

    def _collect_bcb_alternative(self, series_id: str, days_back: int, end_date: datetime) -> Optional[pd.DataFrame]:
        """Coleta dados do BCB com método alternativo"""
        try:
            # Configurações alternativas
            alternative_params = {
                'format': 'json',
                'datatype': 'datetime',
                'convert_dates': True
            }
            
            # Tenta coletar dados com método alternativo
            df = self._collect_bcb_data(series_id, datetime.now() - pd.Timedelta(days=days_back), end_date)
            if df is not None and not df.empty:
                return df
            
            # Se falhar, tenta coletar com parâmetros alternativos
            df = self._collect_bcb_data(series_id, datetime.now() - pd.Timedelta(days=days_back), end_date)
            if df is not None and not df.empty:
                return df
            
            return None
            
        except Exception as e:
            logger.error(f"Erro na coleta alternativa do BCB (série {series_id}): {str(e)}")
            return None

    def _validate_and_process_data(self, df: pd.DataFrame, feature_code: str) -> pd.DataFrame:
        """Valida e processa os dados coletados"""
        try:
            if df.empty:
                return df
            
            # Garante tipos corretos
            df['Date'] = pd.to_datetime(df['Date'])
            df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
            
            # Remove valores nulos e duplicatas
            df = df.dropna()
            df = df.drop_duplicates('Date', keep='last')
            
            # Ordena por data
            df = df.sort_values('Date')
            
            # Remove outliers extremos
            if feature_code not in ['DOLAR', 'SELIC']:
                z_scores = np.abs((df['Value'] - df['Value'].mean()) / df['Value'].std())
                df = df[z_scores < 4]
            
            return df
            
        except Exception as e:
            logger.error(f"Erro na validação dos dados para {feature_code}: {str(e)}")
            return pd.DataFrame()

    def _load_features(self) -> List[Dict[str, Any]]:
        """
        Carrega configurações das features do arquivo JSON
        Returns:
            Lista de configurações das features
        """
        try:
            features_path = os.path.join(self.base_path, 'src', 'data', 'features.json')
            with open(features_path, 'r') as f:
                data = json.load(f)
                features = data.get('features', [])
                
                # Validação adicional das features
                validated_features = []
                for feature in features:
                    # Verifica se tem todos os campos necessários
                    if not all(k in feature for k in ['code', 'source', 'series_id']):
                        logger.warning(f"Feature ignorada por falta de campos obrigatórios: {feature}")
                        continue
                    
                    # Ajusta configurações específicas por fonte
                    if feature['source'] == 'BCB':
                        # Garante que temos o código BCB correto
                        if not feature.get('series_id'):
                            logger.warning(f"Feature BCB ignorada por falta de series_id: {feature}")
                            continue
                        # Adiciona parâmetros específicos do BCB
                        feature['bcb_params'] = {
                            'format': 'json',
                            'datatype': 'datetime',
                            'convert_dates': True
                        }
                    
                    elif feature['source'] == 'YAHOO':
                        # Ajusta período para Yahoo Finance
                        feature['yahoo_params'] = {
                            'interval': '1d',  # Força intervalo diário
                            'period_type': 'days',  # Usa dias ao invés de meses
                            'prepost': False,  # Ignora pre/post market
                            'repair': True  # Tenta reparar dados quebrados
                        }
                    
                    validated_features.append(feature)
                
                if not validated_features:
                    raise ValueError("Nenhuma feature válida encontrada no arquivo de configuração")
                
                logger.info(f"Carregadas {len(validated_features)} features válidas do arquivo de configuração")
                return validated_features
                
        except Exception as e:
            logger.error(f"Erro ao carregar features: {str(e)}")
            return [] 