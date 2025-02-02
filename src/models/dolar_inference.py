from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import json
import logging
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LassoCV
from sklearn.decomposition import PCA
from statsmodels.tsa.stattools import adfuller
from ta.trend import SMAIndicator, EMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.momentum import RSIIndicator, StochasticOscillator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DolarInference:
    """
    Classe para inferência do preço do dólar usando ensemble de modelos
    """
    
    def __init__(self, data: Optional[pd.DataFrame] = None):
        """
        Inicializa o modelo de inferência
        
        Args:
            data: DataFrame com os dados históricos
        """
        self.data = data
        self.target_col = 'DOLAR'  # Nome padrão da coluna alvo
        self.date_col = 'Date'     # Nome padrão da coluna de data
        self.feature_cols: List[str] = []
        self.scaler = RobustScaler()
        self.pca = PCA(n_components=0.95)
        
        # Configurações
        self.required_columns = {
            'target': ['DOLAR', 'dolar', 'USD'],  # Possíveis nomes para coluna alvo
            'date': ['Date', 'DATA', 'date', 'datetime']  # Possíveis nomes para coluna de data
        }
        
        self.models = {
            'rf': RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            'gb': GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            'xgb': XGBRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                min_child_weight=2,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            ),
            'lasso': LassoCV(
                alphas=np.logspace(-4, 1, 50),
                max_iter=2000,
                random_state=42
            )
        }
        self.model_weights = {
            'rf': 0.3,
            'gb': 0.3,
            'xgb': 0.3,
            'lasso': 0.1
        }
        
    def load_data(self) -> None:
        """
        Carrega dados mais recentes com features selecionadas
        """
        try:
            # Encontrar último arquivo de features selecionadas
            features_dir = Path("data/processed/features")
            if not features_dir.exists():
                raise FileNotFoundError(f"Diretório não encontrado: {features_dir}")
                
            data_files = [f for f in features_dir.glob("selected_features_data_*.csv")]
            if not data_files:
                raise FileNotFoundError("Nenhum arquivo de features encontrado")
                
            latest_file = max(data_files, key=lambda x: x.stat().st_mtime)
            logger.info(f"Carregando dados de: {latest_file}")
            
            # Carregar dados
            self.data = pd.read_csv(latest_file)
            if self.data.empty:
                raise ValueError("Arquivo de dados vazio")
            
            # Validar e ajustar nomes das colunas
            self._validate_and_fix_columns()
            
            # Converter data
            self.data[self.date_col] = pd.to_datetime(self.data[self.date_col])
            
            # Ordenar por data
            self.data = self.data.sort_values(self.date_col)
            
            # Identificar features
            self.feature_cols = [col for col in self.data.columns 
                               if col not in [self.date_col, self.target_col]]
            
            logger.info(f"Dados carregados: {len(self.data)} registros, {len(self.feature_cols)} features")
            
        except Exception as e:
            logger.error(f"Erro ao carregar dados: {str(e)}")
            raise

    def _validate_and_fix_columns(self) -> None:
        """
        Valida e corrige nomes das colunas necessárias
        """
        if not isinstance(self.data, pd.DataFrame):
            raise ValueError("DataFrame não inicializado")
            
        # Verificar coluna alvo
        target_found = False
        for possible_name in self.required_columns['target']:
            if possible_name in self.data.columns:
                if possible_name != self.target_col:
                    logger.info(f"Renomeando coluna alvo de '{possible_name}' para '{self.target_col}'")
                    self.data = self.data.rename(columns={possible_name: self.target_col})
                target_found = True
                break
        
        if not target_found:
            raise ValueError(f"Coluna alvo não encontrada. Esperado uma das: {self.required_columns['target']}")
        
        # Verificar coluna de data
        date_found = False
        for possible_name in self.required_columns['date']:
            if possible_name in self.data.columns:
                if possible_name != self.date_col:
                    logger.info(f"Renomeando coluna de data de '{possible_name}' para '{self.date_col}'")
                    self.data = self.data.rename(columns={possible_name: self.date_col})
                date_found = True
                break
        
        if not date_found:
            raise ValueError(f"Coluna de data não encontrada. Esperado uma das: {self.required_columns['date']}")
            
        logger.info("Validação de colunas concluída com sucesso")

    def prepare_data(self) -> None:
        """
        Prepara os dados com feature engineering avançada
        """
        try:
            if not isinstance(self.data, pd.DataFrame) or self.data.empty:
                raise ValueError("Dados não disponíveis")
            
            if self.target_col not in self.data.columns:
                raise ValueError(f"Coluna alvo '{self.target_col}' não encontrada")
            
            logger.info("Iniciando preparação dos dados...")
            
            # Verificar valores ausentes na coluna alvo
            missing_target = self.data[self.target_col].isnull().sum()
            if missing_target > 0:
                logger.warning(f"Encontrados {missing_target} valores ausentes na coluna alvo")
                self.data = self.data.dropna(subset=[self.target_col])
            
            # Verificar valores infinitos
            inf_mask = np.isinf(self.data[self.target_col])
            if inf_mask.any():
                logger.warning(f"Encontrados {inf_mask.sum()} valores infinitos na coluna alvo")
                self.data = self.data[~inf_mask]
            
            # 1. Features técnicas
            for col in [self.target_col]:
                # Tendências
                sma5 = SMAIndicator(close=self.data[col], window=5)
                sma20 = SMAIndicator(close=self.data[col], window=20)
                ema5 = EMAIndicator(close=self.data[col], window=5)
                ema20 = EMAIndicator(close=self.data[col], window=20)
                
                self.data[f'{col}_sma_5'] = sma5.sma_indicator()
                self.data[f'{col}_sma_20'] = sma20.sma_indicator()
                self.data[f'{col}_ema_5'] = ema5.ema_indicator()
                self.data[f'{col}_ema_20'] = ema20.ema_indicator()
                
                # Volatilidade
                bb = BollingerBands(close=self.data[col])
                atr = AverageTrueRange(
                    high=self.data[col].rolling(2).max(),
                    low=self.data[col].rolling(2).min(),
                    close=self.data[col],
                    window=14
                )
                
                self.data[f'{col}_bb_high'] = bb.bollinger_hband()
                self.data[f'{col}_bb_low'] = bb.bollinger_lband()
                self.data[f'{col}_atr'] = atr.average_true_range()
                
                # Momentum
                rsi = RSIIndicator(close=self.data[col])
                stoch = StochasticOscillator(
                    high=self.data[col].rolling(14).max(),
                    low=self.data[col].rolling(14).min(),
                    close=self.data[col]
                )
                
                self.data[f'{col}_rsi'] = rsi.rsi()
                self.data[f'{col}_stoch'] = stoch.stoch()
            
            # 2. Features temporais
            self.data['day_of_week'] = pd.to_datetime(self.data[self.date_col]).dt.dayofweek
            self.data['month'] = pd.to_datetime(self.data[self.date_col]).dt.month
            self.data['quarter'] = pd.to_datetime(self.data[self.date_col]).dt.quarter
            
            # 3. Lags e diferenças
            for lag in [1, 2, 3, 5, 10, 21]:  # Dias úteis: 1 dia, 1 semana, 2 semanas, 1 mês
                self.data[f'{self.target_col}_lag_{lag}'] = self.data[self.target_col].shift(lag)
                self.data[f'{self.target_col}_diff_{lag}'] = self.data[self.target_col].diff(lag)
            
            # 4. Features de volatilidade
            for window in [5, 21]:
                self.data[f'{self.target_col}_volatility_{window}'] = (
                    self.data[self.target_col]
                    .rolling(window=window)
                    .std()
                )
            
            # 5. Características estatísticas
            for window in [5, 21]:
                roll = self.data[self.target_col].rolling(window=window)
                self.data[f'{self.target_col}_kurt_{window}'] = roll.kurt()
                self.data[f'{self.target_col}_skew_{window}'] = roll.skew()
            
            # 6. Remover linhas com NaN
            self.data = self.data.dropna()
            
            # 7. Selecionar features
            self.feature_cols = [col for col in self.data.columns 
                               if col not in [self.target_col, self.date_col]]
            
            # 8. Teste de estacionariedade
            adf_result = adfuller(self.data[self.target_col])
            if adf_result[1] > 0.05:  # Não estacionária
                # Adicionar diferenciação
                self.data[f'{self.target_col}_diff_1'] = self.data[self.target_col].diff()
                self.feature_cols.append(f'{self.target_col}_diff_1')

            logger.info("Preparação dos dados concluída com sucesso")
            
        except Exception as e:
            logger.error(f"Erro ao preparar dados: {str(e)}")
            raise

    def train_and_evaluate(self) -> Dict:
        """
        Treina e avalia modelos usando validação temporal com melhorias
        """
        try:
            if not isinstance(self.data, pd.DataFrame) or self.data.empty:
                raise ValueError("Dados não disponíveis")
            
            if len(self.data) < 30:
                raise ValueError("Quantidade insuficiente de dados para treinamento (mínimo 30 registros)")
            
            X = self.data[self.feature_cols]
            y = self.data[self.target_col]
            
            # Validação temporal adaptativa
            n_splits = min(5, len(X) // 30)
            tscv = TimeSeriesSplit(n_splits=n_splits)
            
            metrics = {
                'mae': [],
                'rmse': [],
                'r2': [],
                'accuracy': [],
                'predictions': []
            }
            
            # Treinar e avaliar modelos
            for train_idx, test_idx in tscv.split(X):
                X_train = X.iloc[train_idx]
                y_train = y.iloc[train_idx]
                X_test = X.iloc[test_idx]
                y_test = y.iloc[test_idx]
                
                # Escalar features
                X_train_scaled = self.scaler.fit_transform(X_train)
                X_test_scaled = self.scaler.transform(X_test)
                
                # Redução de dimensionalidade
                X_train_pca = self.pca.fit_transform(X_train_scaled)
                X_test_pca = self.pca.transform(X_test_scaled)
                
                # Treinar modelos
                predictions = {}
                model_scores = {}
                
                for name, model in self.models.items():
                    # Treinar com dados originais e PCA
                    model.fit(X_train_scaled, y_train)
                    pred_original = model.predict(X_test_scaled)
                    
                    model_pca = self.models[name].__class__(**self.models[name].get_params())
                    model_pca.fit(X_train_pca, y_train)
                    pred_pca = model_pca.predict(X_test_pca)
                    
                    # Combinar previsões
                    pred = (pred_original + pred_pca) / 2
                    predictions[name] = pred.astype(float)
                    
                    # Calcular score para peso dinâmico
                    score = float(r2_score(y_test, pred))
                    model_scores[name] = max(0.1, float(score))  # Mínimo de 0.1
                
                # Atualizar pesos dos modelos
                total_score = sum(model_scores.values())
                self.model_weights = {
                    name: score/total_score 
                    for name, score in model_scores.items()
                }
                
                # Combinar previsões com pesos atualizados
                ensemble_pred = np.zeros(len(y_test))
                for name, pred in predictions.items():
                    ensemble_pred += pred * self.model_weights[name]
                
                # Calcular métricas
                metrics['mae'].append(float(mean_absolute_error(y_test, ensemble_pred)))
                metrics['rmse'].append(float(np.sqrt(mean_squared_error(y_test, ensemble_pred))))
                metrics['r2'].append(float(r2_score(y_test, ensemble_pred)))
                
                # Calcular acurácia direcional
                y_test_values = np.asarray(y_test)
                ensemble_pred_values = np.asarray(ensemble_pred)
                direction_actual = np.sign(np.diff(y_test_values, prepend=y_test_values[0]))
                direction_pred = np.sign(np.diff(ensemble_pred_values, prepend=ensemble_pred_values[0]))
                accuracy = float(np.mean(direction_actual == direction_pred))
                metrics['accuracy'].append(accuracy)
                
                # Salvar previsões
                metrics['predictions'].extend([
                    {
                        'date': date.strftime("%Y-%m-%d"),
                        'actual': float(actual),
                        'predicted': float(pred),
                        'model_weights': self.model_weights.copy()
                    }
                    for date, actual, pred in zip(
                        self.data.iloc[test_idx][self.date_col],
                        y_test,
                        ensemble_pred
                    )
                ])
            
            # Calcular médias
            avg_metrics = {
                'mae': float(np.mean(metrics['mae'])),
                'rmse': float(np.mean(metrics['rmse'])),
                'r2': float(np.mean(metrics['r2'])),
                'accuracy': float(np.mean(metrics['accuracy']))
            }
            
            logger.info("Métricas de avaliação:")
            logger.info(f"MAE: {avg_metrics['mae']:.4f}")
            logger.info(f"RMSE: {avg_metrics['rmse']:.4f}")
            logger.info(f"R²: {avg_metrics['r2']:.4f}")
            logger.info(f"Acurácia Direcional: {avg_metrics['accuracy']:.2%}")
            
            return {
                'metrics': avg_metrics,
                'predictions': metrics['predictions']
            }
            
        except Exception as e:
            logger.error(f"Erro ao treinar e avaliar modelos: {str(e)}")
            raise

    def predict_next_day(self) -> Dict:
        """
        Faz previsão para o próximo dia com intervalo de confiança melhorado
        """
        try:
            if not isinstance(self.data, pd.DataFrame) or self.data.empty:
                raise ValueError("Dados não disponíveis")
            
            if len(self.data) < 21:  # Precisamos de pelo menos 21 dias para calcular volatilidade
                raise ValueError("Histórico insuficiente para previsão (mínimo 21 dias)")
            
            # Preparar dados mais recentes
            last_data = self.data.iloc[-1:][self.feature_cols]
            X_scaled = self.scaler.transform(last_data)
            X_pca = self.pca.transform(X_scaled)
            
            # Fazer previsões com cada modelo
            predictions = {}
            for name, model in self.models.items():
                # Previsão com dados originais
                pred_original = float(model.predict(X_scaled)[0])
                
                # Previsão com PCA
                model_pca = self.models[name].__class__(**self.models[name].get_params())
                model_pca.fit(self.pca.transform(self.scaler.transform(self.data[self.feature_cols])), 
                            self.data[self.target_col])
                pred_pca = float(model_pca.predict(X_pca)[0])
                
                # Média das previsões
                predictions[name] = (pred_original + pred_pca) / 2
            
            # Calcular previsão ensemble
            ensemble_pred = float(sum(
                pred * self.model_weights[name]
                for name, pred in predictions.items()
            ))
            
            # Calcular intervalo de confiança
            recent_volatility = float(self.data[self.target_col].tail(21).std())
            pred_std = float(np.std(list(predictions.values())))
            confidence_width = np.sqrt(pred_std**2 + recent_volatility**2)
            
            confidence_interval = {
                'lower': float(ensemble_pred - 1.96 * confidence_width),
                'upper': float(ensemble_pred + 1.96 * confidence_width)
            }
            
            next_day = self.data[self.date_col].max() + timedelta(days=1)
            
            result = {
                'date': next_day.strftime("%Y-%m-%d"),
                'prediction': ensemble_pred,
                'confidence_interval': confidence_interval,
                'model_predictions': {k: float(v) for k, v in predictions.items()},
                'model_weights': self.model_weights,
                'recent_volatility': recent_volatility
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Erro ao fazer previsão: {str(e)}")
            raise

    def save_results(self, metrics: Dict, prediction: Dict) -> None:
        """
        Salva resultados da inferência
        
        Args:
            metrics: Métricas de avaliação
            prediction: Previsão para próximo dia
        """
        try:
            # Criar diretório se não existir
            output_dir = Path("data/processed/inference")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Gerar timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            
            # Salvar previsões históricas
            predictions_df = pd.DataFrame(metrics['predictions'])
            predictions_df['date'] = pd.to_datetime(predictions_df['date'])
            predictions_df = predictions_df.sort_values('date')
            
            pred_path = output_dir / f"historical_predictions_{timestamp}.csv"
            predictions_df.to_csv(pred_path, index=False)
            
            # Salvar previsão para próximo dia
            next_day_df = pd.DataFrame([{
                'date': prediction['date'],
                'predicted_dolar': prediction['prediction'],
                'confidence_lower': prediction['confidence_interval']['lower'],
                'confidence_upper': prediction['confidence_interval']['upper']
            }])
            
            next_day_path = output_dir / f"next_day_prediction_{timestamp}.csv"
            next_day_df.to_csv(next_day_path, index=False)
            
            # Salvar metadados
            metadata = {
                'timestamp': timestamp,
                'metrics': metrics['metrics'],
                'features': self.feature_cols,
                'model_weights': self.model_weights,
                'next_day_prediction': prediction
            }
            
            metadata_path = output_dir / f"inference_metadata_{timestamp}.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
                
            logger.info(f"Resultados salvos em: {output_dir}")
            
        except Exception as e:
            logger.error(f"Erro ao salvar resultados: {str(e)}")
            raise

def main():
    """
    Executa inferência do dólar
    """
    try:
        # Inicializar inferência
        inference = DolarInference()
        
        # Carregar e preparar dados
        inference.load_data()
        inference.prepare_data()
        
        # Treinar e avaliar modelos
        metrics = inference.train_and_evaluate()
        
        # Fazer previsão para próximo dia
        prediction = inference.predict_next_day()
        
        # Salvar resultados
        inference.save_results(metrics, prediction)
        
        # Exibir previsão
        print("\nPrevisão para o próximo dia:")
        print(f"Data: {prediction['date']}")
        print(f"Dólar previsto: R$ {prediction['prediction']:.4f}")
        print(f"Intervalo de confiança: R$ {prediction['confidence_interval']['lower']:.4f} - R$ {prediction['confidence_interval']['upper']:.4f}")
        
    except Exception as e:
        logger.error(f"Erro na inferência: {str(e)}")
        raise

if __name__ == "__main__":
    main() 