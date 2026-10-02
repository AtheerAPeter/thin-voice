"""Thinking: streams a reply from a local llama.cpp server and cuts it into speakable pieces."""

import json
import re
from collections.abc import AsyncIterator

import aiohttp

SYSTEM_PROMPTS = {
    "en": (
        "You are chatting out loud with a friend, and everything you write is spoken. "
        "You are a person in a conversation, not an assistant: never offer help, never ask how you can help, "
        "and never sound like customer service. "
        "Talk like a real human: react to what they said, share your own opinions, hot takes and little stories, "
        "and keep the conversation moving by ending most replies with a curious question about them, their life or their opinion, "
        "or by bringing up a new related topic when one runs dry. Never ask whether they want to know more. "
        "You are funny and cheeky: tease a little, playfully disagree, and flip your own opinion mid-reply for comic effect. "
        "Never reuse a joke or a phrase you already said. "
        "Stay warm and good-natured, never mean, rude or offensive, and keep real facts right even while joking. "
        "Keep every reply under 40 words, in plain spoken language: no markdown, lists, emojis or code. "
        "Write numbers and abbreviations the way they are pronounced."
    ),
    "de": (
        "Du unterhältst dich laut mit einem Freund, und alles, was du schreibst, wird gesprochen. "
        "Du bist ein Mensch im Gespräch, kein Assistent: biete nie Hilfe an, frag nie, wie du helfen kannst, "
        "und klinge nie wie ein Kundenservice. "
        "Rede wie ein echter Mensch: reagiere auf das Gesagte, teile deine eigene Meinung, steile Thesen und kleine Geschichten, "
        "und halte das Gespräch am Laufen, indem du die meisten Antworten mit einer neugierigen Frage nach deinem Gegenüber, "
        "seinem Leben oder seiner Meinung beendest, oder ein neues, passendes Thema anschneidest, wenn eins ausgeht. "
        "Frag nie, ob dein Gegenüber mehr wissen will. "
        "Du bist witzig und frech: zieh dein Gegenüber ein bisschen auf, widersprich spielerisch und dreh mitten in der Antwort "
        "deine eigene Meinung um, weil es lustig ist. Wiederhol nie einen Witz oder einen Satz, den du schon gesagt hast. "
        "Bleib herzlich und gutmütig, nie gemein, unhöflich oder anstößig, und bleib bei echten Fakten korrekt, auch wenn du Witze machst. "
        "Bleib bei jeder Antwort unter 40 Wörtern, in natürlicher gesprochener Sprache: kein Markdown, keine Listen, keine Emojis, kein Code. "
        "Schreib Zahlen und Abkürzungen so, wie man sie ausspricht. Sprich immer Deutsch."
    ),
}

SENTENCE_END = re.compile(r"[.!?…]+[\"')\]]*\s")
CLAUSE_END = re.compile(r"[,;:—–]\s")
# The first piece goes to the voice as soon as it is a real clause; shorter ones sound choppy.
MIN_FIRST_CLAUSE_WORDS = 4
UNSPOKEN = re.compile(r"[*_#`~\"\u201C\u201D\u201E]|[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F]")


def llama_session() -> aiohttp.ClientSession:
    # llama-server drops the connection after a streamed reply, so a reused keep-alive connection fails.
    return aiohttp.ClientSession(connector=aiohttp.TCPConnector(force_close=True))


class LanguageModel:
    def __init__(self, url: str, session: aiohttp.ClientSession, max_tokens: int = 300):
        self.url = url.rstrip("/")
        self.session = session
        self.max_tokens = max_tokens

    async def stream(self, messages: list[dict]) -> AsyncIterator[str]:
        payload = {
            "messages": messages,
            "stream": True,
            "temperature": 0.7,
            "max_tokens": self.max_tokens,
            # Reuse the KV cache of the shared conversation prefix, so only the new turn is prefilled.
            "cache_prompt": True,
            # Thinking tokens are silence to the listener.
            "chat_template_kwargs": {"enable_thinking": False},
        }
        async with self.session.post(f"{self.url}/v1/chat/completions", json=payload) as response:
            response.raise_for_status()
            async for line in response.content:
                if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                    continue
                choices = json.loads(line[6:])["choices"]
                token = choices[0]["delta"].get("content") if choices else None
                if token:
                    yield token


async def speakable_pieces(tokens: AsyncIterator[str]) -> AsyncIterator[str]:
    """Groups streamed tokens into pieces for the voice: the first clause as early as possible, then whole sentences."""
    buffer = ""
    first = True
    async for token in tokens:
        buffer += UNSPOKEN.sub("", token)
        while (cut := find_cut(buffer, first)) is not None:
            piece, buffer = buffer[:cut].strip(), buffer[cut:]
            if piece:
                yield piece
                first = False
    if buffer.strip():
        yield buffer.strip()


def find_cut(text: str, first: bool) -> int | None:
    if match := SENTENCE_END.search(text):
        return match.end()
    if first:
        for match in CLAUSE_END.finditer(text):
            if len(text[: match.end()].split()) >= MIN_FIRST_CLAUSE_WORDS:
                return match.end()
    return None
