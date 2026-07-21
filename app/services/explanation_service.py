import json
from typing import Any

from app.clients.ollama_client import OllamaClient
from app.core.config import settings
from app.schemas.requests import CountryEvidence, ExplainRequest
from app.schemas.responses import ExplainResponse

# System prompts per output language. Portuguese is the workspace default; any
# unknown language code falls back to the English instructions but still asks
# the model to write in that language (see _language_directive).
_SYSTEM_PROMPT_PT = """Você é um analista de entrada em mercados (market-entry) que ajuda um negócio \
a decidir EM QUAIS PAÍSES vender um produto, a partir de um "pacote de evidências" já calculado \
pelo motor determinístico de MOS (Market Opportunity Score) da MarketMind.

REGRAS INVIOLÁVEIS:
- Use SOMENTE os números fornecidos no pacote. NUNCA invente, estime, arredonde de forma nova \
ou recalcule qualquer valor. Não escreva nenhum número que não esteja no input.
- Você EXPLICA os resultados determinísticos; você não os recalcula nem os contesta.
- Inclua TODOS os países do pacote (não omita nenhum), na ordem do MOS (maior primeiro). \
Cite o MOS de cada país EXATAMENTE como está no input — o ranking já é final, não reordene \
nem recalcule por conta própria.

PRODUZA, nesta ordem:
1) Uma recomendação curta de "onde vender", com os países ranqueados do melhor para o pior, \
citando o MOS de cada um.
2) Um parágrafo curto por país explicando o seu MOS a partir dos seus sinais (signals) e \
dimensões (dimensions).

DISTINÇÃO CRÍTICA (torne-a explícita usando o sinal de volume de demanda, que traz buscas \
absolutas + população + penetração por 100 mil habitantes):
- Separe "vende muito principalmente porque o país é populoso" (volume alto, mas penetração BAIXA) \
de "vende muito porque uma parcela alta das pessoas realmente quer o produto" (penetração ALTA = \
apetite real).
- Deixe claro que um mercado menor, mas com penetração/qualidade alta, ainda pode ser uma forte \
oportunidade.
- Cite concretamente os números reais (buscas absolutas, população, penetração por 100 mil).

Seja conciso e com tom de negócio."""

_SYSTEM_PROMPT_EN = """You are a market-entry analyst helping a business decide IN WHICH COUNTRIES \
to sell a product, based on an "evidence packet" already computed by MarketMind's deterministic \
MOS (Market Opportunity Score) engine.

INVIOLABLE RULES:
- Use ONLY the numbers provided in the packet. NEVER invent, estimate, re-round or recompute any \
value. Do not write any number that is not in the input.
- You EXPLAIN the deterministic results; you do not recompute or dispute them.
- Include EVERY country in the packet (omit none), ordered by MOS (highest first). Quote each \
country's MOS EXACTLY as given in the input — the ranking is already final, do not reorder or \
recompute it yourself.

PRODUCE, in this order:
1) A short "where to sell" recommendation, with the countries ranked best to worst, quoting each \
country's MOS.
2) One short paragraph per country explaining its MOS from its signals and dimensions.

CRITICAL DISTINCTION (make it explicit using the demand-volume signal, which carries absolute \
searches + population + penetration per 100k inhabitants):
- Separate "sells a lot mainly because the country is populous" (high volume but LOW penetration) \
from "sells a lot because a high share of people genuinely want it" (high penetration = real \
appetite).
- Make clear that a smaller market with high penetration/quality can still be a strong opportunity.
- Concretely reference the real figures (absolute searches, population, penetration per 100k).

Be concise and business-toned."""


def _language_directive(language: str) -> str:
    lang = (language or "").strip().lower()
    if lang in ("pt", "pt-br", "pt_br", "portuguese", "portugues", "português"):
        return _SYSTEM_PROMPT_PT
    if lang in ("en", "en-us", "en_us", "english"):
        return _SYSTEM_PROMPT_EN
    # Unknown code: use English instructions but force the requested language.
    return f'{_SYSTEM_PROMPT_EN}\n\nWrite the entire narrative in this language: "{language}".'


def build_system_prompt(language: str) -> str:
    return _language_directive(language)


def _country_payload(country: CountryEvidence) -> dict[str, Any]:
    return {
        "country": country.country,
        "mos": country.mos,
        "mos_band": country.mos_band,
        "dimensions": country.dimensions,
        "signals": [
            {
                "code": s.code,
                "label": s.label,
                "score": s.score,
                "band": s.band,
                "meaning": s.meaning,
                "detail": s.detail,
            }
            for s in country.signals
        ],
    }


def build_user_message(request: ExplainRequest) -> str:
    """Render the evidence packet as a compact, clearly-labelled JSON block."""
    packet = {
        "product_name": request.product_name,
        "language": request.language,
        "countries": [_country_payload(c) for c in request.countries],
    }
    packet_json = json.dumps(packet, ensure_ascii=False, indent=2)
    return (
        f"Produto avaliado / Product evaluated: {request.product_name}\n\n"
        "Pacote de evidências MOS (não altere nenhum número) / "
        "MOS evidence packet (do not alter any number):\n\n"
        f"{packet_json}"
    )


class ExplanationService:
    def __init__(self, client: OllamaClient | None = None) -> None:
        self._client = client or OllamaClient()

    async def explain(self, request: ExplainRequest) -> ExplainResponse:
        system_prompt = build_system_prompt(request.language)
        user_message = build_user_message(request)

        narrative = await self._client.chat(system_prompt, user_message)

        return ExplainResponse(
            narrative=narrative,
            model=settings.ollama_model,
            product_name=request.product_name,
        )
