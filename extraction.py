import os
import re
import time

try:
    from google.api_core import exceptions as google_exceptions
except ImportError:

    class google_exceptions:
        class ResourceExhausted(Exception):
            pass

        class TooManyRequests(Exception):
            pass
from pydantic import BaseModel, Field

from model_client import create_genai_client


class Entity(BaseModel):
    name: str = Field(description="Entity name in Title Case")
    type: str = Field(description="One of: PERSON, ORGANIZATION, LOCATION, EVENT, WORK")
    description: str = Field(description="One sentence describing this entity in context")


class Triple(BaseModel):
    subject: str
    relation: str
    object: str


class ExtractionResult(BaseModel):
    entities: list[Entity] = Field(description="Named entities with type and description")
    triples: list[Triple] = Field(description="Relationships between extracted entities")


EXTRACTION_PROMPT = """You are a Knowledge Graph Specialist. Extract named entities and relationships from the text.

## Entity Types (ONLY extract these)
- PERSON: Individuals with proper names (e.g., "Albert Einstein", "Marie Curie")
- ORGANIZATION: Companies, institutions, teams, bands (e.g., "NASA", "The Beatles")
- LOCATION: Geographic places with proper names (e.g., "Tokyo", "Mount Everest")
- EVENT: Named events, wars, competitions (e.g., "World War II", "2008 Summer Olympics")
- WORK: Creative works, publications, products (e.g., "Harry Potter", "The New York Times")

## DO NOT Extract
- Pronouns or references: he, she, they, it, this, that, the former, the latter
- Generic terms: the company, the author, a person, the team, the city
- Partial dates: '47, the 90s, last year (only named dates like "March 2020")
- Bare numbers: 100, 37 (unless part of a name like "Apollo 11")
- Common nouns without proper names: city, country, book, song, album, film
- Adjectives or descriptors: famous, large, important, American

## Critical Rules
- Use the FULL proper name: "Albert Einstein" not "Einstein", "United States" not "US"
- Use third person: name the subject explicitly, never use pronouns
- Each triple MUST reference entities from the entities list
- In JSON triples, use exactly these keys: subject, relation, object. Never use predicate.
- Extract at most 12 entities and 15 triples. Prefer entities and relations central to the document title.
- Keep every entity description to 20 words or fewer.
- Input format is: first line = document title (**title**), next lines = paragraph text - consider both when extracting
- You MUST consider BOTH title and paragraph text when extracting entities, relations, and descriptions
- Consider the FULL text when writing descriptions

## Example
Text: "Einstein developed relativity while at the Swiss Patent Office in Bern."
Entities:
- Albert Einstein | PERSON | Physicist who developed the theory of relativity
- Swiss Patent Office | ORGANIZATION | Government office where Einstein worked
- Bern | LOCATION | Swiss city where the Patent Office is located

Triples:
- (Albert Einstein, worked at, Swiss Patent Office)
- (Swiss Patent Office, located in, Bern)

Text:
{text}"""


class TripleExtractor:
    def __init__(
        self,
        model_id: str = "gemini-2.5-flash",
        max_retries: int = 5,
        base_delay: float = 2.0,
    ) -> None:
        self._client = create_genai_client()
        self._model_id = model_id
        self._max_retries = max_retries
        self._base_delay = base_delay

    def extract(self, text: str) -> ExtractionResult | None:
        prompt = EXTRACTION_PROMPT.format(text=text)

        for attempt in range(self._max_retries):
            try:
                response = self._client.models.generate_content(
                    model=self._model_id,
                    contents=prompt,
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": ExtractionResult,
                        "temperature": 0.2,
                        "max_output_tokens": int(os.getenv("QASA_EXTRACTION_MAX_OUTPUT_TOKENS", "768")),
                    },
                )

                result = response.parsed or self._parse_text_result(response.text or "")
                if result is None:
                    print(
                        f"[Extraction] Unparseable response (attempt {attempt + 1}/{self._max_retries}); retrying",
                        flush=True,
                    )
                    if attempt + 1 < self._max_retries:
                        time.sleep(self._base_delay * (2**attempt))
                        continue
                    return None

                return self._normalize_result(result)
            except (google_exceptions.ResourceExhausted, google_exceptions.TooManyRequests):
                delay = self._base_delay * (2**attempt)
                print(f"[Throttled] Waiting {delay:.1f}s (attempt {attempt + 1})")
                time.sleep(delay)
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    delay = self._base_delay * (2**attempt)
                    print(f"[Throttled] Waiting {delay:.1f}s (attempt {attempt + 1})")
                    time.sleep(delay)
                else:
                    raise

        raise RuntimeError(f"Failed to extract after {self._max_retries} retries")


    def _parse_text_result(self, text: str) -> ExtractionResult | None:
        entities: list[Entity] = []
        triples: list[Triple] = []
        section = None

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            lower = line.lower().rstrip(':')
            if lower in {"entities", "entity"}:
                section = "entities"
                continue
            if lower in {"triples", "relationships", "relations"}:
                section = "triples"
                continue

            line = re.sub(r"^[-*\d.)\s]+", "", line).strip()
            if section == "entities" and "|" in line:
                parts = [part.strip() for part in line.split("|", 2)]
                if len(parts) == 3 and parts[0]:
                    entities.append(Entity(name=parts[0], type=parts[1] or "WORK", description=parts[2]))
            elif section == "triples":
                match = re.match(r"^\(?\s*([^,]+?)\s*,\s*([^,]+?)\s*,\s*(.+?)\s*\)?$", line)
                if match:
                    triples.append(Triple(subject=match.group(1), relation=match.group(2), object=match.group(3)))

        if not entities and not triples:
            return None
        return ExtractionResult(entities=entities, triples=triples)

    def _normalize_result(self, result: ExtractionResult) -> ExtractionResult:
        return ExtractionResult(
            entities=[
                Entity(
                    name=self._normalize(e.name),
                    type=e.type.upper().strip(),
                    description=e.description.strip(),
                )
                for e in result.entities
                if e.name.strip()
            ],
            triples=[
                Triple(
                    subject=self._normalize(t.subject),
                    relation=t.relation.strip().lower(),
                    object=self._normalize(t.object),
                )
                for t in result.triples
            ],
        )

    def _normalize(self, text: str) -> str:
        return " ".join(text.strip().lower().split())
