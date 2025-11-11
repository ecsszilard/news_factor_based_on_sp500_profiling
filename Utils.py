import numpy as np
import pandas as pd
import time
import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class NewsType(Enum):
    """Hírtípusok váratlanság és hatás szerint"""
    RUMOR = "rumor"  # Pletyka - előre jelzi a jövőbeli hírt
    EXPECTED = "expected"  # Várt hír - már beárazott
    SURPRISE = "surprise"  # Meglepetés - váratlan esemény
    CONFIRMING = "confirming"  # Megerősítő - korábbi pletykát validál
    CONTEXT_DEPENDENT = "context"  # Kontextus-függő - más hírrel együtt értékes

class NewsScope(Enum):
    """Hír hatáskörének mértéke"""
    COMPANY_SPECIFIC = "company"  # Cégspecifikus (1-2 cég)
    SECTOR = "sector"  # Szektorális (azonos szektorú cégek)
    MARKET_WIDE = "market"  # Piacszintű (FED, makro hírek)

@dataclass
class NewsEvent:
    """Híresemény teljes metaadatokkal"""
    text: str
    timestamp: float
    news_type: NewsType
    news_scope: NewsScope
    
    # Direkt hatásvektor h ∈ R^N
    direct_impact_vector: np.ndarray  # Melyik cégre mennyire hat közvetlenül
    
    impact_magnitude: float  # Alaphatás nagysága
    related_news_ids: List[int]  # Kapcsolódó hírek ID-i
    priced_in_factor: float  # Mennyire van beárazva (0-1)
    requires_context: bool  # Más hír kell az értelmezéshez
    decay_rate: float  # Hatás csökkenési üteme
    lambda_spillover: float  # Spillover erősítési paraméter λ
    keywords: List[str]
    news_id: int
    
    @property
    def affected_companies(self) -> List[int]:
        """Mely cégindexek érintettek közvetlenül (h_i != 0)"""
        return np.where(self.direct_impact_vector != 0)[0].tolist()

class Utils:
    
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)
        self.news_counter = 0
        self.news_graph = {}  # Hírek közötti kapcsolatok
    
    def create_hybrid_data(self, 
                          num_companies: int = 20,
                          num_news: int = 150,
                          days: int = 90,
                          train_ratio: float = 0.7):
        """
        Hibrid adatgenerálás: realisztikus ár- és hírgenerálás fejlett szimulációval
        
        Args:
            num_companies: Cégek száma
            num_news: Hírek teljes száma
            days: Napok száma
            train_ratio: Train/val arány
            
        Returns:
            companies_df, train_news, val_news, sample_prices, correlation_matrix, covariance_matrix
        """
        # 1. Cégek és szektorok létrehozása
        companies_df, sector_map = self._create_companies(num_companies)
        company_symbols = companies_df['symbol'].tolist()
        
        # 2. Korrelációs mátrix C és kovarianciamátrix Σ építése
        correlation_matrix = self._build_correlation_matrix(company_symbols, sector_map)
        volatilities = np.random.uniform(0.015, 0.035, size=num_companies)
        covariance_matrix = self._build_covariance_matrix(volatilities, correlation_matrix)
        
        # 3. Időbeli paraméterek
        base_date = datetime.datetime.now() - datetime.timedelta(days=days)
        train_days = int(days * train_ratio)
        val_start_date = base_date + datetime.timedelta(days=train_days)
        end_date = base_date + datetime.timedelta(days=days)
        
        # 4. Hírek generálása kapcsolati gráffal
        all_news_events = self._generate_news_with_relationships(
            num_news=num_news,
            num_companies=num_companies,
            companies=company_symbols,
            sector_map=sector_map,
            start_date=base_date,
            end_date=end_date
        )
        
        # 5. Train/val split időbeli alapon
        train_news_events = [n for n in all_news_events if n.timestamp < val_start_date.timestamp()]
        val_news_events = [n for n in all_news_events if n.timestamp >= val_start_date.timestamp()]
        
        # 6. Árfolyamok generálása hírekhez és korrelációhoz igazodva
        sample_prices = self._generate_prices_with_news_impact(
            companies=company_symbols,
            news_events=all_news_events,
            covariance_matrix=covariance_matrix,
            base_date=base_date,
            days=days
        )
        
        train_news = [self._convert_event_to_dict(n, company_symbols) for n in train_news_events]
        val_news = [self._convert_event_to_dict(n, company_symbols) for n in val_news_events]
        return companies_df, train_news, val_news, sample_prices, correlation_matrix, covariance_matrix
    
    def _create_companies(self, num_companies: int) -> Tuple[pd.DataFrame, Dict]:
        """Cégek és szektorok generálása"""
        all_companies = [
            ('AAPL', 'Apple Inc.', 'Technology'),
            ('MSFT', 'Microsoft Corp.', 'Technology'),
            ('NVDA', 'NVIDIA Corp.', 'Technology'),
            ('GOOGL', 'Alphabet Inc.', 'Technology'),
            ('META', 'Meta Platforms', 'Technology'),
            ('AMZN', 'Amazon.com Inc.', 'Consumer Discretionary'),
            ('TSLA', 'Tesla Inc.', 'Consumer Discretionary'),
            ('NIKE', 'Nike Inc.', 'Consumer Discretionary'),
            ('HD', 'Home Depot Inc.', 'Consumer Discretionary'),
            ('WMT', 'Walmart Inc.', 'Consumer Staples'),
            ('PG', 'Procter & Gamble', 'Consumer Staples'),
            ('F', 'Ford Motor Co.', 'Consumer Discretionary'),
            ('GM', 'General Motors', 'Consumer Discretionary'),
            ('JPM', 'JPMorgan Chase', 'Financials'),
            ('BAC', 'Bank of America', 'Financials'),
            ('GS', 'Goldman Sachs', 'Financials'),
            ('XOM', 'Exxon Mobil', 'Energy'),
            ('CVX', 'Chevron Corp.', 'Energy'),
            ('PFE', 'Pfizer Inc.', 'Healthcare'),
            ('JNJ', 'Johnson & Johnson', 'Healthcare'),
            ('UNH', 'UnitedHealth Group', 'Healthcare'),
            ('NFLX', 'Netflix Inc.', 'Communication Services'),
            ('DIS', 'Walt Disney Co.', 'Communication Services'),
            ('T', 'AT&T Inc.', 'Communication Services'),
        ][:num_companies]
        
        companies_df = pd.DataFrame(all_companies, columns=['symbol', 'name', 'sector'])
        sector_map = dict(zip(companies_df['symbol'], companies_df['sector']))
        
        return companies_df, sector_map
    
    def _build_correlation_matrix(self, companies: List[str], sector_map: Dict) -> np.ndarray:
        """
        Korrelációs mátrix C építése:
        - Azonos szektor: 0.5-0.8
        - Különböző szektor: 0.1-0.3
        - Speciális párok: magasabb korreláció
        """
        n = len(companies)
        corr_matrix = np.eye(n)
        
        # Speciális párok (erősebb kapcsolat)
        special_pairs = {
            ('AAPL', 'MSFT'): 0.75,
            ('F', 'GM'): 0.85,
            ('JPM', 'BAC'): 0.80,
            ('XOM', 'CVX'): 0.82,
            ('PFE', 'JNJ'): 0.70,
            ('TSLA', 'NVDA'): 0.65,
        }
        
        for i, c1 in enumerate(companies):
            for j, c2 in enumerate(companies):
                if i >= j:
                    continue
                
                # Ellenőrizzük speciális párokat
                pair_key = tuple(sorted([c1, c2]))
                if pair_key in special_pairs or (c2, c1) in special_pairs:
                    corr = special_pairs.get(pair_key, special_pairs.get((c2, c1)))
                else:
                    # Szektor alapú korreláció
                    s1, s2 = sector_map.get(c1, 'Unknown'), sector_map.get(c2, 'Unknown')
                    if s1 == s2:
                        corr = np.random.uniform(0.5, 0.8)
                    else:
                        corr = np.random.uniform(0.1, 0.3)
                
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr
        
        return corr_matrix
    
    def _build_covariance_matrix(self, volatilities: np.ndarray, correlation_matrix: np.ndarray) -> np.ndarray:
        """
        Kovarianciamátrix Σ építése:
        Σ_ij = σ_i * σ_j * C_ij
        """
        return np.outer(volatilities, volatilities) * correlation_matrix
    
    def _generate_news_with_relationships(self,
                                         num_news: int,
                                         num_companies: int,
                                         companies: List[str],
                                         sector_map: Dict,
                                         start_date: datetime.datetime,
                                         end_date: datetime.datetime) -> List[NewsEvent]:
        """
        Hírek generálása direkt hatásvektorral h ∈ R^N
        """
        news_events = []
        date_range_seconds = (end_date - start_date).total_seconds()
        
        # Szektor → cégindex mapping
        sector_to_companies = {}
        for i, company in enumerate(companies):
            sector = sector_map.get(company, 'Unknown')
            if sector not in sector_to_companies:
                sector_to_companies[sector] = []
            sector_to_companies[sector].append(i)
        
        # Hírsablonok scopeonként és típusonként
        # Formátum: (template, impact_magnitude, priced_in_factor, decay_rate, lambda_spillover)
        
        company_templates = {
            NewsType.RUMOR: [
                ("Sources suggest {company} may be planning major restructuring", 0.01, 0.3, 0.5, 0.1),
                ("Unconfirmed reports of {company} acquisition talks", 0.015, 0.4, 0.5, 0.15),
                ("Industry insiders hint at {company} product delays", -0.01, 0.3, 0.5, 0.1),
            ],
            NewsType.EXPECTED: [
                ("{company} announces expected quarterly earnings in line with forecasts", 0.005, 0.9, 0.3, 0.05),
                ("{company} confirms previously announced dividend payment", 0.002, 0.95, 0.2, 0.03),
            ],
            NewsType.SURPRISE: [
                ("{company} reports unexpected 30% earnings beat", 0.04, 0.0, 0.4, 0.2),
                ("{company} CEO unexpectedly resigns effective immediately", -0.03, 0.0, 0.4, 0.15),
                ("{company} faces surprise regulatory investigation", -0.035, 0.05, 0.5, 0.2),
            ],
            NewsType.CONFIRMING: [
                ("{company} officially confirms earlier merger speculation", 0.008, 0.7, 0.3, 0.1),
                ("{company} validates rumored partnership with tech giant", 0.012, 0.6, 0.3, 0.12),
            ],
            NewsType.CONTEXT_DEPENDENT: [
                ("{company} reports supply chain improvements", 0.015, 0.3, 0.6, 0.15),
                ("{company} announces cost reduction initiative", 0.01, 0.4, 0.5, 0.1),
            ]
        }
        
        sector_templates = [
            ("Technology sector faces increased regulatory pressure on AI", ['Technology'], -0.02, 0.2, 0.4, 0.3),
            ("Automotive industry supply chain shows recovery signs", ['Consumer Discretionary'], 0.025, 0.3, 0.4, 0.25),
            ("Energy sector benefits from rising commodity prices", ['Energy'], 0.03, 0.2, 0.3, 0.35),
            ("Financial sector impacted by interest rate changes", ['Financials'], 0.015, 0.4, 0.5, 0.3),
        ]
        
        market_templates = [
            ("Federal Reserve raises interest rates by 0.75% to combat inflation", 0.02, 0.3, 0.5, 0.5),
            ("Fed announces quantitative tightening, reducing liquidity", -0.025, 0.2, 0.6, 0.6),
            ("Global economic growth forecast revised downward amid recession fears", -0.03, 0.1, 0.5, 0.55),
            ("Central banks worldwide coordinate monetary policy to stabilize markets", 0.015, 0.4, 0.4, 0.45),
            ("Trade tensions escalate between major economies", -0.02, 0.2, 0.5, 0.5),
            ("Unexpected geopolitical crisis triggers market volatility", -0.035, 0.0, 0.6, 0.65),
        ]
        
        # 1. Hírláncolatok (pletyka → confirming)
        num_chains = int(num_news * 0.15)
        
        for _ in range(num_chains):
            company_idx = np.random.randint(num_companies)
            company = companies[company_idx]
            
            # Pletyka
            rumor_template = company_templates[NewsType.RUMOR][
                np.random.randint(len(company_templates[NewsType.RUMOR]))
            ]
            text, impact, priced, decay, lambda_sp = rumor_template
            
            # h vektor: csak ez az egy cég
            h_rumor = np.zeros(num_companies)
            h_rumor[company_idx] = impact
            
            rumor_time = start_date + datetime.timedelta(
                seconds=np.random.uniform(0, date_range_seconds * 0.7)
            )
            
            rumor_news = NewsEvent(
                text=text.format(company=company),
                timestamp=rumor_time.timestamp(),
                news_type=NewsType.RUMOR,
                news_scope=NewsScope.COMPANY_SPECIFIC,
                direct_impact_vector=h_rumor,
                impact_magnitude=impact,
                related_news_ids=[],
                priced_in_factor=priced,
                requires_context=False,
                decay_rate=decay,
                lambda_spillover=lambda_sp,
                keywords=['rumor', 'speculation'],
                news_id=self.news_counter
            )
            news_events.append(rumor_news)
            rumor_id = self.news_counter
            self.news_counter += 1
            
            # Megerősítő hír
            confirm_time = rumor_time + datetime.timedelta(days=np.random.uniform(3, 10))
            if confirm_time < end_date:
                confirm_template = company_templates[NewsType.CONFIRMING][
                    np.random.randint(len(company_templates[NewsType.CONFIRMING]))
                ]
                text_c, impact_c, priced_c, decay_c, lambda_c = confirm_template
                
                h_confirm = np.zeros(num_companies)
                h_confirm[company_idx] = impact_c
                
                confirm_news = NewsEvent(
                    text=text_c.format(company=company),
                    timestamp=confirm_time.timestamp(),
                    news_type=NewsType.CONFIRMING,
                    news_scope=NewsScope.COMPANY_SPECIFIC,
                    direct_impact_vector=h_confirm,
                    impact_magnitude=impact_c,
                    related_news_ids=[rumor_id],
                    priced_in_factor=priced_c,
                    requires_context=False,
                    decay_rate=decay_c,
                    lambda_spillover=lambda_c,
                    keywords=['confirmed', 'official'],
                    news_id=self.news_counter
                )
                news_events.append(confirm_news)
                self.news_counter += 1
        
        # 2. Szektorális hírek
        num_sector = int(num_news * 0.2)
        for _ in range(num_sector):
            template = sector_templates[int(np.random.randint(len(sector_templates)))]
            text, affected_sectors, impact, priced, decay, lambda_sp = template
            
            # h vektor: minden érintett szektorban azonos hatás
            h_sector = np.zeros(num_companies)
            for sector in affected_sectors:
                if sector in sector_to_companies:
                    for idx in sector_to_companies[sector]:
                        h_sector[idx] = impact
            
            news_time = start_date + datetime.timedelta(seconds=np.random.uniform(0, date_range_seconds))
            
            sector_news = NewsEvent(
                text=text,
                timestamp=news_time.timestamp(),
                news_type=NewsType.SURPRISE,
                news_scope=NewsScope.SECTOR,
                direct_impact_vector=h_sector,
                impact_magnitude=impact,
                related_news_ids=[],
                priced_in_factor=priced,
                requires_context=False,
                decay_rate=decay,
                lambda_spillover=lambda_sp,
                keywords=['sector', 'industry'],
                news_id=self.news_counter
            )
            news_events.append(sector_news)
            self.news_counter += 1
        
        # 3. Piacszintű hírek (FED, makro) - h = (1,1,...,1)^T
        num_market = int(num_news * 0.15)
        for _ in range(num_market):
            template = market_templates[int(np.random.randint(len(market_templates)))]
            text, impact, priced, decay, lambda_sp = template
            
            # h vektor: mindenki egyformán érintett
            h_market = np.ones(num_companies) * impact
            
            news_time = start_date + datetime.timedelta(seconds=np.random.uniform(0, date_range_seconds))
            
            market_news = NewsEvent(
                text=text,
                timestamp=news_time.timestamp(),
                news_type=NewsType.SURPRISE,
                news_scope=NewsScope.MARKET_WIDE,
                direct_impact_vector=h_market,
                impact_magnitude=impact,
                related_news_ids=[],
                priced_in_factor=priced,
                requires_context=False,
                decay_rate=decay,
                lambda_spillover=lambda_sp,
                keywords=['fed', 'macro', 'market'],
                news_id=self.news_counter
            )
            news_events.append(market_news)
            self.news_counter += 1
        
        # 4. Kontextus-függő párok
        num_context = int(num_news * 0.1)
        for _ in range(num_context):
            company_idx = np.random.randint(num_companies)
            company = companies[company_idx]
            
            template1 = company_templates[NewsType.CONTEXT_DEPENDENT][
                np.random.randint(len(company_templates[NewsType.CONTEXT_DEPENDENT]))
            ]
            text1, impact1, priced1, decay1, lambda1 = template1
            
            h1 = np.zeros(num_companies)
            h1[company_idx] = impact1 * 0.3  # Egyedül kisebb hatás
            
            time1 = start_date + datetime.timedelta(seconds=np.random.uniform(0, date_range_seconds * 0.8))
            
            news1 = NewsEvent(
                text=text1.format(company=company),
                timestamp=time1.timestamp(),
                news_type=NewsType.CONTEXT_DEPENDENT,
                news_scope=NewsScope.COMPANY_SPECIFIC,
                direct_impact_vector=h1,
                impact_magnitude=impact1 * 0.3,
                related_news_ids=[],
                priced_in_factor=priced1,
                requires_context=True,
                decay_rate=decay1,
                lambda_spillover=lambda1,
                keywords=['context'],
                news_id=self.news_counter
            )
            news_events.append(news1)
            news1_id = self.news_counter
            self.news_counter += 1
            
            # Második hír
            time2 = time1 + datetime.timedelta(days=np.random.uniform(1, 5))
            if time2 < end_date:
                h2 = np.zeros(num_companies)
                h2[company_idx] = impact1  # Együtt teljes hatás
                
                news2 = NewsEvent(
                    text=f"{company} reveals strategic rationale behind recent initiatives",
                    timestamp=time2.timestamp(),
                    news_type=NewsType.CONTEXT_DEPENDENT,
                    news_scope=NewsScope.COMPANY_SPECIFIC,
                    direct_impact_vector=h2,
                    impact_magnitude=impact1,
                    related_news_ids=[news1_id],
                    priced_in_factor=0.3,
                    requires_context=True,
                    decay_rate=0.3,
                    lambda_spillover=lambda1,
                    keywords=['strategy', 'context'],
                    news_id=self.news_counter
                )
                news_events.append(news2)
                self.news_counter += 1
        
        # 5. Többi: vegyes cégspecifikus hírek
        remaining = num_news - len(news_events)
        for _ in range(remaining):
            news_type = np.random.choice([NewsType.EXPECTED, NewsType.SURPRISE], p=[0.4, 0.6])
            template = company_templates[news_type][
                np.random.randint(len(company_templates[news_type]))
            ]
            text, impact, priced, decay, lambda_sp = template
            
            company_idx = np.random.randint(num_companies)
            company = companies[company_idx]
            
            h = np.zeros(num_companies)
            h[company_idx] = impact
            
            news_time = start_date + datetime.timedelta(seconds=np.random.uniform(0, date_range_seconds))
            
            news = NewsEvent(
                text=text.format(company=company),
                timestamp=news_time.timestamp(),
                news_type=news_type,
                news_scope=NewsScope.COMPANY_SPECIFIC,
                direct_impact_vector=h,
                impact_magnitude=impact,
                related_news_ids=[],
                priced_in_factor=priced,
                requires_context=False,
                decay_rate=decay,
                lambda_spillover=lambda_sp,
                keywords=[news_type.value],
                news_id=self.news_counter
            )
            news_events.append(news)
            self.news_counter += 1
        
        # Időrendi rendezés
        news_events.sort(key=lambda x: x.timestamp)
        return news_events
    
    def _generate_prices_with_news_impact(self,
                                         companies: List[str],
                                         news_events: List[NewsEvent],
                                         covariance_matrix: np.ndarray,
                                         base_date: datetime.datetime,
                                         days: int) -> Dict[str, Dict]:
        """
        Árfolyamok generálása a képlet szerint:
        Δr = h + λ * Σ * h
        
        ahol:
        - Δr: árfolyamváltozás vektor
        - h: direkt hírhatás vektor
        - λ: spillover erősítési paraméter
        - Σ: kovarianciamátrix
        """
        n_companies = len(companies)
        
        # Napi alapreturns (korrelált Gauss-zajból)
        base_returns = np.random.multivariate_normal(
            np.zeros(n_companies), 
            covariance_matrix, 
            size=days
        )
        
        # Hírek hatásának számítása: Δr = h + λ * Σ * h
        news_impact = np.zeros((days, n_companies))
        
        for news in news_events:
            news_date = datetime.datetime.fromtimestamp(news.timestamp)
            news_day = (news_date - base_date).days
            
            if news_day < 0 or news_day >= days:
                continue
            
            # Direkt hatásvektor h
            h = news.direct_impact_vector.copy()
            
            # Beárazás figyelembevétele
            actual_impact_factor = (1 - news.priced_in_factor)
            h = h * actual_impact_factor
            
            # Kontextus-függő: csak ha van kapcsolódó hír
            if news.requires_context and news.related_news_ids:
                h = h * 1.5
            
            # Spillover hatás: λ * Σ * h
            spillover = news.lambda_spillover * (covariance_matrix @ h)
            
            # Teljes hatás: Δr = h + spillover
            total_impact = h + spillover
            
            # Exponenciális decay alkalmazása napokra
            for day_offset in range(min(7, days - news_day)):
                decay_factor = np.exp(-day_offset * news.decay_rate)
                day_idx = news_day + day_offset
                
                if day_idx < days:
                    news_impact[day_idx, :] += total_impact * decay_factor
        
        # Végső hozamok
        final_returns = base_returns + news_impact
        
        # Árak számítása
        price_data = {}
        for i, company in enumerate(companies):
            price = np.random.uniform(50, 300)
            prices = {}
            
            for day in range(days):
                price *= (1 + final_returns[day, i])
                price = max(price, 1.0)
                
                day_date = base_date + datetime.timedelta(days=day)
                prices[day_date.timestamp()] = price
            
            price_data[company] = prices
        
        return price_data
    
    def _convert_event_to_dict(self, news: NewsEvent, companies: List[str]) -> Dict:
        """Helper to convert NewsEvent object to dict for the processor."""

        return {
            'text': news.text,
            'timestamp': news.timestamp,
            'companies': companies if news.news_scope == NewsScope.MARKET_WIDE else [companies[i] for i in news.affected_companies],
            'news_type': news.news_type.value,
            'news_scope': news.news_scope.value,
            'impact_magnitude': news.impact_magnitude,
            'priced_in_factor': news.priced_in_factor,
            'lambda_spillover': news.lambda_spillover,
            'direct_impact_vector': news.direct_impact_vector,
        }