import numpy as np
import time
from collections import defaultdict
import logging

logger = logging.getLogger("AdvancedNewsFactor.FinancialImpactTracker")

class FinancialImpactTracker:
    """
    Követi a szavak historikus pénzügyi hatását cégekre
    """
    
    def __init__(self):
        # word -> company -> [price_impacts]
        self.word_company_impacts = defaultdict(lambda: defaultdict(list))
        # company -> word -> average_impact
        self.company_word_sensitivity = defaultdict(dict)
        
    def update_impact(self, words, company, price_change_1d, price_change_5d):
        """
        Frissíti egy szó finanszírozott hatását
        """
        impact_score = abs(price_change_1d) * 0.7 + abs(price_change_5d) * 0.3
        
        for word in words:
            self.word_company_impacts[word][company].append({
                'impact': impact_score,
                'direction': 1 if price_change_1d > 0 else -1,
                'timestamp': time.time()
            })
            
            # Frissítjük az átlagos érzékenységet
            impacts = self.word_company_impacts[word][company]
            if len(impacts) > 0:
                avg_impact = np.mean([imp['impact'] for imp in impacts[-10:]])  # utolsó 10
                self.company_word_sensitivity[company][word] = avg_impact

    def get_financial_attention_weights(self, words, company, normalize=True):
        """
        Pénzügyi attention súlyok lekérése szavakhoz és céghez
        
        ParamÉterek:
            words (list): Szavak listája
            company (str): Cég szimbólum
            normalize (bool): Normalizáljuk-e a súlyokat
            
        VisszatÉrÉsi ÉrtÉk:
            numpy.ndarray: Financial attention súlyok
        """
        weights = []
        
        for word in words:
            if company in self.company_word_sensitivity and word in self.company_word_sensitivity[company]:
                weight = self.company_word_sensitivity[company][word]
            else:
                weight = 0.1  # Alapértelmezett kis súly ismeretlen szavakhoz
            weights.append(weight)
        
        weights = np.array(weights)
        
        if normalize and len(weights) > 0:
            # Softmax normalizálás
            weights = np.exp(weights - np.max(weights))  # Numerikus stabilitás
            weights = weights / (np.sum(weights) + 1e-8)
        
        return weights