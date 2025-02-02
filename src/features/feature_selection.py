import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, List, Union, Any
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler, RobustScaler
import xgboost as xgb
import logging
import json
from scipy import stats
import os
from sklearn.feature_selection import SelectKBest, mutual_info_regression
import glob

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureSelector:
    def __init__(self, lookback_days: int = 365):
        """
        Inicializa o seletor de features
        Args:
            lookback_days: Número de dias para análise retroativa
        """
        self.lookback_days = lookback_days
        self.base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.data_dir = os.path.join(self.base_path, 'data', 'processed')
        self.news_dir = os.path.join(self.base_path, 'data', 'news')
        
        # Configura logging
        self._setup_logging()

    def _setup_logging(self):
        """Configura o sistema de logging"""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(level=logging.INFO, format=log_format)

    def _load_data(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Carrega dados históricos e de notícias
        Returns:
            Tuple[DataFrame com dados históricos, DataFrame com notícias]
        """
        try:
            # Carrega dados históricos
            historical_file = os.path.join(self.data_dir, 'historical_data.csv')
            if not os.path.exists(historical_file):
                logger.error("Arquivo de dados históricos não encontrado")
                return None, None
            
            df = pd.read_csv(historical_file)
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Filtra pelo período de lookback
            end_date = df['Date'].max()
            start_date = end_date - timedelta(days=self.lookback_days)
            df = df[df['Date'] >= start_date]
            
            # Carrega notícias
            news_df = None
            news_file = os.path.join(self.news_dir, 'historical_news.csv')
            if os.path.exists(news_file):
                news_df = pd.read_csv(news_file)
                news_df['date'] = pd.to_datetime(news_df['date'])
                news_df = news_df[news_df['date'] >= start_date]
            
            return df, news_df
            
        except Exception as e:
            logger.error(f"Erro ao carregar dados: {str(e)}")
            return None, None

    def _calculate_news_impact(self, df: pd.DataFrame, news_df: Optional[pd.DataFrame]) -> pd.DataFrame:
        """
        Calcula métricas de impacto das notícias
        Args:
            df: DataFrame com dados históricos
            news_df: DataFrame com notícias
        Returns:
            DataFrame com métricas de notícias
        """
        if news_df is None:
            return df
        
        try:
            # Agrupa notícias por data
            daily_news = news_df.groupby('date').agg({
                'news_id': 'count',
                'source': lambda x: len(set(x))
            }).rename(columns={
                'news_id': 'news_count',
                'source': 'source_count'
            })
            
            # Adiciona métricas de notícias aos dados históricos
            df = df.merge(
                daily_news,
                left_on='Date',
                right_index=True,
                how='left'
            )
            
            # Preenche dias sem notícias com 0
            df[['news_count', 'source_count']] = df[['news_count', 'source_count']].fillna(0)
            
            return df
            
        except Exception as e:
            logger.error(f"Erro ao calcular impacto das notícias: {str(e)}")
            return df

    def _prepare_features(self, df: pd.DataFrame, target_col: str = 'DOLAR') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepara features para seleção de forma otimizada, usando apenas dados disponíveis
        """
        try:
            if df.empty:
                logger.warning("DataFrame de entrada está vazio")
                return pd.DataFrame(), pd.Series()

            if target_col not in df.columns:
                logger.error(f"Coluna alvo '{target_col}' não encontrada")
                return pd.DataFrame(), pd.Series()

            # Remove colunas não numéricas exceto Date
            date_col = df['Date'].copy()
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df_numeric = df[numeric_cols].copy()
            
            # Verifica se há dados suficientes
            if len(df_numeric.columns) < 2:  # Precisa de pelo menos target + 1 feature
                logger.error("Dados insuficientes para análise")
                return pd.DataFrame(), pd.Series()
            
            # Prepara dicionário para armazenar features válidas
            feature_dict = {}
            
            # Processa cada coluna com validação
            for col in df_numeric.columns:
                if col != target_col:
                    try:
                        # Calcula features básicas primeiro
                        series = df_numeric[col].copy()
                        if series.isna().all():
                            continue
                            
                        base_features = {
                            f'{col}_return': series.pct_change(),
                            f'{col}_volatility': series.rolling(window=10, min_periods=1).std()
                        }
                        
                        # Adiciona médias móveis se houver dados suficientes
                        if len(series.dropna()) >= 20:
                            ma_features = {
                                f'{col}_MA5': series.rolling(window=5, min_periods=1).mean().pct_change(),
                                f'{col}_MA10': series.rolling(window=10, min_periods=1).mean().pct_change(),
                                f'{col}_MA20': series.rolling(window=20, min_periods=1).mean().pct_change()
                            }
                            base_features.update(ma_features)
                        
                        # Calcula RSI apenas se houver dados suficientes
                        if len(series.dropna()) >= 14:
                            delta = series.diff()
                            gain = delta.mask(delta < 0, 0).rolling(window=14, min_periods=1).mean()
                            loss = (-delta).mask(delta > 0, 0).rolling(window=14, min_periods=1).mean()
                            rs = gain / loss
                            base_features[f'{col}_rsi'] = 100 - (100 / (1 + rs))
                        
                        # Filtra features válidas
                        valid_features = {k: v for k, v in base_features.items() 
                                       if not v.isna().all() and not np.isinf(v).any()}
                        
                        feature_dict.update(valid_features)
                        
                    except Exception as e:
                        logger.warning(f"Erro ao processar coluna {col}: {str(e)}")
                        continue
            
            if not feature_dict:
                logger.error("Nenhuma feature válida gerada")
                return pd.DataFrame(), pd.Series()
            
            # Cria DataFrame de features
            features_df = pd.DataFrame(feature_dict, index=df.index)
            
            # Limpa dados
            features_df = features_df.replace([np.inf, -np.inf], np.nan)
            features_df = features_df.dropna(how='all', axis=1)  # Remove colunas totalmente vazias
            features_df = features_df.fillna(method='ffill').fillna(method='bfill')  # Preenche valores faltantes
            
            if features_df.empty:
                logger.warning("DataFrame de features vazio após limpeza")
                return pd.DataFrame(), pd.Series()
            
            # Prepara target com validação
            target_series = df[target_col].pct_change()
            target_series = target_series.loc[features_df.index]
            target_series = target_series.fillna(method='ffill')
            
            if target_series.isna().any():
                logger.warning("Valores ausentes na série target após processamento")
                valid_idx = ~target_series.isna()
                features_df = features_df.loc[valid_idx]
                target_series = target_series.loc[valid_idx]
            
            # Normaliza features
            try:
                scaler = RobustScaler()
                X_scaled = pd.DataFrame(
                    scaler.fit_transform(features_df),
                    index=features_df.index,
                    columns=features_df.columns
                )
            except Exception as e:
                logger.warning(f"Erro na normalização, usando dados originais: {str(e)}")
                X_scaled = features_df
            
            return X_scaled, target_series
            
        except Exception as e:
            logger.error(f"Erro na preparação das features: {str(e)}")
            return pd.DataFrame(), pd.Series()

    def _get_latest_news_impact_file(self) -> Optional[str]:
        """
        Encontra o arquivo mais recente de news_impact_summary
        Returns:
            Path do arquivo mais recente ou None se não encontrar
        """
        try:
            # Lista todos os arquivos que começam com news_impact_summary_
            files = glob.glob(os.path.join(self.data_dir, 'news_impact_summary_*.csv'))
            
            if not files:
                logger.warning("Nenhum arquivo news_impact_summary encontrado")
                return None
                
            # Ordena por data de modificação e pega o mais recente
            latest_file = max(files, key=os.path.getmtime)
            logger.info(f"Usando arquivo de impacto de notícias: {latest_file}")
            return latest_file
            
        except Exception as e:
            logger.error(f"Erro ao buscar arquivo de impacto de notícias: {str(e)}")
            return None

    def select_features(self, k: int = 10, target_col: str = 'DOLAR') -> Dict[str, Any]:
        """
        Seleciona as features mais relevantes
        Args:
            k: Número de features a selecionar
            target_col: Nome da coluna alvo
        Returns:
            Dict com resultados da seleção
        """
        try:
            # Carrega dados históricos
            df, _ = self._load_data()
            if df is None:
                logger.error("Falha ao carregar dados históricos")
                return {}
            
            logger.info(f"Dados históricos carregados: {len(df)} registros")
            logger.info(f"Colunas disponíveis: {df.columns.tolist()}")
            
            # Carrega dados de notícias do último arquivo de impacto
            news_impact_file = self._get_latest_news_impact_file()
            if news_impact_file:
                try:
                    news_impact_df = pd.read_csv(news_impact_file)
                    news_impact_df['date'] = pd.to_datetime(news_impact_df['date'])
                    
                    # Merge com dados históricos
                    df = df.merge(
                        news_impact_df,
                        left_on='Date',
                        right_on='date',
                        how='left'
                    )
                    
                    # Remove colunas duplicadas e desnecessárias
                    columns_to_drop = ['date']
                    if 'target_return' in df.columns:
                        columns_to_drop.append('target_return')
                    df = df.drop(columns=columns_to_drop)
                    
                    # Preenche dias sem notícias com 0
                    news_columns = ['news_count', 'source_count']
                    df[news_columns] = df[news_columns].fillna(0)
                    
                    logger.info("Dados de notícias incorporados com sucesso")
                except Exception as e:
                    logger.error(f"Erro ao carregar dados de notícias: {str(e)}")
                    # Remove colunas de notícias se existirem
                    news_columns = ['news_count', 'source_count']
                    df = df.drop(columns=[col for col in news_columns if col in df.columns])
            
            # Prepara features
            X, y = self._prepare_features(df, target_col)
            if X.empty or y.empty:
                logger.error("Falha na preparação das features")
                return {}
            
            # Ajusta k se necessário
            k = min(k, len(X.columns))
            if k < 1:
                logger.error("Número insuficiente de features para seleção")
                return {}
            
            # Seleciona features usando múltiplos métodos
            results = {
                'timestamp': datetime.now().strftime('%Y%m%d_%H%M'),
                'target_variable': target_col,
                'lookback_days': self.lookback_days,
                'feature_importance': {}
            }
            
            # 1. Informação Mútua
            mi_selector = SelectKBest(score_func=mutual_info_regression, k=k)
            mi_selector.fit(X, y)
            mi_scores = pd.DataFrame({
                'feature': X.columns,
                'mutual_info_score': mi_selector.scores_,
                'selected_mi': mi_selector.get_support()
            })
            
            # 2. Random Forest Importance
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X, y)
            rf_scores = pd.DataFrame({
                'feature': X.columns,
                'rf_importance': rf.feature_importances_
            })
            
            # 3. Lasso Importance
            lasso = LassoCV(cv=5, random_state=42)
            lasso.fit(X, y)
            lasso_scores = pd.DataFrame({
                'feature': X.columns,
                'lasso_coef': np.abs(lasso.coef_)
            })
            
            # 4. XGBoost Importance
            xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
            xgb_model.fit(X, y)
            xgb_scores = pd.DataFrame({
                'feature': X.columns,
                'xgb_importance': xgb_model.feature_importances_
            })
            
            # Combina scores
            feature_scores = mi_scores.merge(rf_scores, on='feature')
            feature_scores = feature_scores.merge(lasso_scores, on='feature')
            feature_scores = feature_scores.merge(xgb_scores, on='feature')
            
            # Normaliza scores
            for col in ['mutual_info_score', 'rf_importance', 'lasso_coef', 'xgb_importance']:
                feature_scores[f'{col}_norm'] = feature_scores[col] / feature_scores[col].max()
            
            # Calcula score combinado
            feature_scores['combined_score'] = (
                feature_scores['mutual_info_score_norm'] +
                feature_scores['rf_importance_norm'] +
                feature_scores['lasso_coef_norm'] +
                feature_scores['xgb_importance_norm']
            ) / 4
            
            # Seleciona top k features
            selected_features = feature_scores.nlargest(k, 'combined_score')
            
            # Calcula correlações
            correlations = X[selected_features['feature']].corrwith(y)
            
            # Prepara resultados finais
            results.update({
                'selected_features': selected_features['feature'].tolist(),
                'feature_scores': feature_scores.to_dict('records'),
                'correlations': correlations.to_dict(),
                'data_range': {
                    'start': df['Date'].min().strftime('%Y-%m-%d'),
                    'end': df['Date'].max().strftime('%Y-%m-%d')
                },
                'statistics': {
                    'total_features': len(X.columns),
                    'total_samples': len(y),
                    'features_selected': k
                }
            })
            
            # Adiciona estatísticas de notícias se disponíveis
            if news_impact_file and 'news_count' in df.columns:
                results['news_statistics'] = {
                    'total_news': int(df['news_count'].sum()),
                    'unique_sources': int(df['source_count'].sum()),
                    'avg_daily_news': float(df['news_count'].mean()),
                    'max_daily_news': int(df['news_count'].max())
                }
            
            # Salva resultados
            timestamp = datetime.now().strftime('%Y%m%d_%H%M')
            results_file = os.path.join(self.data_dir, f'feature_selection_{timestamp}.json')
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Seleção de features concluída. Resultados salvos em {results_file}")
            return results
            
        except Exception as e:
            logger.error(f"Erro na seleção de features: {str(e)}")
            return {}

def main():
    """
    Executa seleção de features e salva resultados
    """
    try:
        # Inicializar seletor
        selector = FeatureSelector(lookback_days=365)
        
        # Executar seleção de features
        results = selector.select_features(k=10, target_col='DOLAR')
        
        if not results:
            logger.error("Falha na seleção de features")
            return
        
        # Exibir resultados
        print("\nFeatures Selecionadas:")
        for i, feature in enumerate(results['selected_features'], 1):
            # Encontra os scores para a feature atual
            feature_data = next(s for s in results['feature_scores'] if s['feature'] == feature)
            combined_score = feature_data.get('combined_score', 0.0)
            correlation = results['correlations'].get(feature, 'N/A')
            
            print(f"{i}. {feature}")
            print(f"   Score Combinado: {combined_score:.4f}")
            print(f"   Correlação: {correlation if isinstance(correlation, str) else correlation:.4f}")
            
            # Exibe scores individuais se disponíveis
            if 'mutual_info_score' in feature_data:
                print(f"   Mutual Info Score: {feature_data['mutual_info_score']:.4f}")
            if 'rf_importance' in feature_data:
                print(f"   Random Forest Importance: {feature_data['rf_importance']:.4f}")
            if 'xgb_importance' in feature_data:
                print(f"   XGBoost Importance: {feature_data['xgb_importance']:.4f}")
            print()
        
        print("\nEstatísticas:")
        stats = results.get('statistics', {})
        print(f"Total de features analisadas: {stats.get('total_features', 'N/A')}")
        print(f"Total de amostras: {stats.get('total_samples', 'N/A')}")
        
        data_range = results.get('data_range', {})
        if data_range:
            print(f"Período: {data_range.get('start', 'N/A')} a {data_range.get('end', 'N/A')}")
        
        # Exibe estatísticas de notícias se disponíveis
        news_stats = results.get('news_statistics', {})
        if news_stats:
            print("\nEstatísticas de Notícias:")
            print(f"Total de notícias: {news_stats.get('total_news', 'N/A')}")
            print(f"Fontes únicas: {news_stats.get('unique_sources', 'N/A')}")
            print(f"Média diária: {news_stats.get('avg_daily_news', 'N/A'):.2f}")
        
        # Exibe arquivos gerados se disponíveis
        output_files = results.get('output_files', {})
        if output_files:
            print("\nArquivos gerados:")
            for file_type, file_path in output_files.items():
                print(f"- {file_type}: {file_path}")
        
    except Exception as e:
        logger.error(f"Erro na execução: {str(e)}")
        return None

if __name__ == "__main__":
    main() 