from pydantic import BaseModel, Field


class SignalEvidence(BaseModel):
    code: str
    label: str
    score: float | None = None
    band: str | None = None
    meaning: str | None = None
    # The detail carries the concrete numbers, e.g. for demand volume:
    # "1526130 total monthly searches ...; population 212,812,405;
    #  penetration 717.1 searches per 100k inhabitants ...".
    detail: str | None = None


class CountryEvidence(BaseModel):
    country: str  # ISO3 code
    mos: float
    mos_band: str
    dimensions: dict[str, float] = Field(default_factory=dict)  # dimension sub-scores 0-100
    signals: list[SignalEvidence] = Field(default_factory=list)


class ExplainRequest(BaseModel):
    product_name: str
    language: str = "pt"  # output language of the narrative
    countries: list[CountryEvidence] = Field(min_length=1)
