from app.clients.ollama_client import strip_think


def test_strip_single_think_block() -> None:
    raw = "<think>o usuário quer...\nvou raciocinar</think>Recomendação: BRA é o melhor mercado."
    assert strip_think(raw) == "Recomendação: BRA é o melhor mercado."


def test_strip_multiline_think_dotall() -> None:
    raw = "<think>\nlinha 1\nlinha 2\n</think>\n\nNarrativa final."
    assert strip_think(raw) == "Narrativa final."


def test_strip_multiple_think_blocks() -> None:
    raw = "<think>a</think>Primeiro.<think>b</think> Segundo."
    assert strip_think(raw) == "Primeiro. Segundo."


def test_no_think_block_is_trimmed_only() -> None:
    assert strip_think("  Sem raciocínio.  ") == "Sem raciocínio."
