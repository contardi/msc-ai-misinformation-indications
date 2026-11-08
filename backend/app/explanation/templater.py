"""Turn the score + token attributions into a plain-language Portuguese explanation."""

from __future__ import annotations


def build_explanation(prob: float, top_tokens: list[dict]) -> str:
    """Compose the user-facing explanation string."""
    pct = round(prob * 100)

    terms = [t["token"] for t in top_tokens[:5] if t.get("token")]
    terms_clause = (
        f" Os termos que mais influenciaram esta estimativa foram: {', '.join(terms)}."
        if terms
        else ""
    )

    return (
        f"O modelo estima {pct}% de probabilidade de o texto conter desinformação. "
        "Trata-se de uma estimativa probabilística, não de um veredito sobre a veracidade do "
        f"conteúdo.{terms_clause}"
    )
