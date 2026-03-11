from .equity import EQUITY_TERMS_EN, EQUITY_TERMS_ES, EQUITY_TERMS_PT
from .indigeneous import INDIGENOUS_COMMUNITY_TERMS
from .reviews import REVIEW_KEYWORDS
from .mental_health import MENTAL_HEALTH_TERMS

# TODO: helpers to convert into compiled regex and bulk-apply
# copy-paste code should be in these notebooks: https://gitlab.pik-potsdam.de/mcc-apsis/living-evidence-maps/lancet-countdown/-/tree/main/2025?ref_type=heads

__all__ = [
    'EQUITY_TERMS_PT',
    'EQUITY_TERMS_ES',
    'EQUITY_TERMS_EN',
    'INDIGENOUS_COMMUNITY_TERMS',
    'REVIEW_KEYWORDS',
    'MENTAL_HEALTH_TERMS',
]
