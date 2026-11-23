"""spaCy-only candidate generation; candidates are never graph facts."""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class SpacyRuntimeUnavailable(RuntimeError):
    """Raised when spaCy's native runtime cannot be imported."""


class SpacyCandidateExtractor:
    """Run sentence, token, NER, dependency, and optional noun-phrase analysis."""

    def __init__(
        self,
        *,
        english_model: str = "en_core_web_sm",
        multilingual_parser_model: str = "xx_sent_ud_sm",
        multilingual_ner_model: str = "xx_ent_wiki_sm",
    ) -> None:
        self.english_model = english_model
        self.multilingual_parser_model = multilingual_parser_model
        self.multilingual_ner_model = multilingual_ner_model
        self._spacy_module: Any = None
        self._spacy_import_error: Exception | None = None
        self._warned_capabilities: set[str] = set()

    def _get_spacy(self) -> Any:
        if self._spacy_module is not None:
            return self._spacy_module
        if self._spacy_import_error is not None:
            raise SpacyRuntimeUnavailable(
                "spaCy's native runtime is unavailable"
            ) from self._spacy_import_error
        try:
            import spacy
        except (ImportError, OSError) as exc:
            self._spacy_import_error = exc
            logger.warning(
                "spaCy could not load; continuing with lexical fallback candidates: %s",
                exc,
            )
            raise SpacyRuntimeUnavailable(
                "spaCy's native runtime is unavailable"
            ) from exc
        self._spacy_module = spacy
        return spacy

    @lru_cache(maxsize=4)
    def _load_model(self, name: str) -> Any:
        try:
            return self._get_spacy().load(name)
        except OSError as exc:
            raise RuntimeError(
                f"Required spaCy model {name!r} is unavailable. "
                "Install project dependencies before running ingestion."
            ) from exc

    def _tokenizer_model_for(self, language: str) -> Any:
        code = (language or "und").split("-", 1)[0].casefold()
        return self._load_model(
            self.english_model if code == "en" else self.multilingual_parser_model
        )

    def _models_for(self, language: str) -> tuple[Any, Any]:
        tokenizer_model = self._tokenizer_model_for(language)
        if "ner" in tokenizer_model.pipe_names:
            return tokenizer_model, tokenizer_model
        try:
            ner_model = self._load_model(self.multilingual_ner_model)
        except RuntimeError as exc:
            if isinstance(exc, SpacyRuntimeUnavailable):
                raise
            self._warn_fallback("NER", self.multilingual_ner_model)
            ner_model = None
        return tokenizer_model, ner_model

    def _warn_fallback(self, capability: str, model_name: str) -> None:
        if capability in self._warned_capabilities:
            return
        self._warned_capabilities.add(capability)
        logger.warning(
            "spaCy %s unavailable in model %s; using lexical fallback for that capability",
            capability,
            model_name,
        )

    def _candidate_rows(
        self,
        doc: Any,
        *,
        has_sentence_segmentation: bool,
        has_dependency_parser: bool,
        model_name: str,
    ) -> Dict[str, Any]:
        if has_sentence_segmentation:
            try:
                sentences = [
                    sentence.text.strip()
                    for sentence in doc.sents
                    if sentence.text.strip()
                ]
            except (ValueError, NotImplementedError):
                self._warn_fallback("sentence segmentation", model_name)
                sentences = [
                    sentence.strip()
                    for sentence in re.split(r"(?<=[.!?؟。])\s+|\n+", doc.text)
                    if sentence.strip()
                ]
        else:
            sentences = [
                sentence.strip()
                for sentence in re.split(r"(?<=[.!?؟。])\s+|\n+", doc.text)
                if sentence.strip()
            ]
        tokens = [token.text for token in doc if not token.is_space]
        dependencies = (
            [
                {
                    "token": token.text,
                    "dependency": token.dep_,
                    "head": token.head.text,
                }
                for token in doc
                if not token.is_space
            ]
            if has_dependency_parser
            else []
        )
        noun_phrases = []
        if has_dependency_parser:
            try:
                noun_phrases = [
                    phrase.text.strip()
                    for phrase in doc.noun_chunks
                    if phrase.text.strip()
                ]
            except (ValueError, NotImplementedError):
                self._warn_fallback("noun phrase extraction", model_name)
        return {
            "sentences": sentences,
            "tokens": tokens,
            "dependencies": dependencies,
            "noun_phrases": noun_phrases,
        }

    def extract(self, text: str, language: str = "und") -> Dict[str, Any]:
        if not text.strip():
            return {"sentences": [], "tokens": [], "dependencies": [], "entities": [], "noun_phrases": []}
        try:
            tokenizer_model, ner_model = self._models_for(language)
        except SpacyRuntimeUnavailable:
            return self._lexical_fallback(text)
        parsed = tokenizer_model(text)
        pipe_names = set(tokenizer_model.pipe_names)
        has_sentence_segmentation = bool(
            {"parser", "senter", "sentencizer"} & pipe_names
        )
        has_dependency_parser = "parser" in pipe_names
        if not has_sentence_segmentation:
            self._warn_fallback("sentence segmentation", tokenizer_model.meta.get("name", "unknown"))
        if not has_dependency_parser:
            self._warn_fallback("dependency parsing", tokenizer_model.meta.get("name", "unknown"))
        candidates = self._candidate_rows(
            parsed,
            has_sentence_segmentation=has_sentence_segmentation,
            has_dependency_parser=has_dependency_parser,
            model_name=tokenizer_model.meta.get("name", "unknown"),
        )
        ner_pipe_names = set(ner_model.pipe_names) if ner_model is not None else set()
        if "ner" not in ner_pipe_names:
            self._warn_fallback(
                "NER",
                ner_model.meta.get("name", "unknown") if ner_model is not None
                else self.multilingual_ner_model,
            )
            candidates["entities"] = []
            candidates["ner_model"] = ""
        else:
            ner_doc = ner_model(text)
            candidates["entities"] = [
                {
                    "text": entity.text,
                    "label": entity.label_,
                    "start": entity.start_char,
                    "end": entity.end_char,
                }
                for entity in ner_doc.ents
            ]
            candidates["ner_model"] = ner_model.meta.get("name", "")
        candidates["parser_model"] = tokenizer_model.meta.get("name", "")
        logger.debug(
            "Generated spaCy candidate hints: language=%s sentences=%d entities=%d",
            language,
            len(candidates["sentences"]),
            len(candidates["entities"]),
        )
        return candidates

    def token_count(self, text: str, language: str = "und") -> int:
        try:
            tokenizer_model = self._tokenizer_model_for(language)
        except SpacyRuntimeUnavailable:
            return sum(not character.isspace() for character in text)
        return len(tokenizer_model.make_doc(text))

    @staticmethod
    def _lexical_fallback(text: str) -> Dict[str, Any]:
        return {
            "sentences": [
                sentence.strip()
                for sentence in re.split(r"(?<=[.!?؟。])\s+|\n+", text)
                if sentence.strip()
            ],
            "tokens": re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE),
            "dependencies": [],
            "entities": [],
            "noun_phrases": [],
            "parser_model": "",
            "ner_model": "",
        }


spacy_candidate_extractor = SpacyCandidateExtractor()
