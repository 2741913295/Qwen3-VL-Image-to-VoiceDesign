"""Validated voice fingerprints and deterministic Chinese instruction rendering."""

from __future__ import annotations

import re
import random
from dataclasses import asdict, dataclass, replace
from difflib import SequenceMatcher
from typing import Any, Iterable, Mapping


AGE_ANCHORS = (8, 12, 15, 18, 20, 22, 23, 25, 28, 30, 32, 35, 38, 40, 42, 45, 50, 55, 60, 65, 70)
MATURE_ADULT_AGE = 32
VOICE_PRESENTATIONS = ("feminine", "masculine", "androgynous")
PITCH_CENTERS = ("very_low", "low", "medium_low", "medium", "medium_high", "high")
PITCH_RANGES = ("narrow", "moderate", "wide")
VOCAL_WEIGHTS = ("very_light", "light", "medium", "full", "heavy")
BRIGHTNESSES = ("dark", "slightly_dark", "neutral", "bright", "very_bright")
RESONANCE_POSITIONS = ("head", "front", "mixed", "chest")
RESONANCE_DEPTHS = ("shallow", "medium", "deep")
BREATHINESS_LEVELS = ("none", "slight", "medium", "strong")
NASALITY_LEVELS = ("none", "slight", "medium", "strong")
TEXTURES = ("clean", "silky", "airy", "husky", "grainy", "metallic", "crisp")
ROUGHNESS_LEVELS = ("none", "slight", "medium", "strong")
ONSETS = ("soft", "balanced", "firm")
ARTICULATIONS = ("soft", "natural", "crisp", "firm")
VOCAL_TENSIONS = ("relaxed", "neutral", "tense")
WARMTH_LEVELS = ("cool", "neutral", "warm")
VOCAL_DISTANCES = ("intimate", "natural", "projected")
MATURITY_LEVELS = ("youthful", "young", "mature")
VOCAL_MATURITIES = (
    "youthful",
    "young_adult",
    "mature_young",
    "mature",
    "seasoned",
)
VISUAL_MASSES = ("delicate", "light", "balanced", "solid", "imposing")
FACIAL_CONTOURS = ("soft", "balanced", "angular", "broad")
BODY_SILHOUETTES = ("slender", "balanced", "broad")
STYLING_STRUCTURES = ("fluid", "structured", "armored")
FACIAL_MATURITIES = ("youthful", "young", "mature", "aged")
FACIAL_HAIR_LEVELS = ("none", "light", "full")
CLOTHING_WEIGHTS = ("light", "medium", "heavy")
POSTURES = ("relaxed", "composed", "upright", "dynamic", "commanding")
EXPRESSIONS = ("gentle", "neutral", "serious", "stern", "lively")
ANCHOR_CONFIDENCES = ("low", "medium", "high")
BASE_SPEECH_RATES = ("slow", "medium_slow", "medium", "medium_fast", "fast")
DYNAMIC_RANGES = ("narrow", "moderate", "wide")
PAUSE_PATTERNS = ("sparse", "regular", "deliberate", "long", "abrupt")
RHYTHM_STYLES = ("even", "clipped", "flowing", "elastic", "syncopated")
BEHAVIOR_ARTICULATIONS = ("soft", "natural", "precise", "crisp", "firm", "rounded")
EMOTIONAL_REACTIVITIES = ("very_low", "low", "medium", "high", "very_high")
RESTRAINT_STYLES = ("inward", "restrained", "balanced", "expressive", "explosive")
SENTENCE_ENERGIES = ("gentle", "steady", "driving", "forceful")
ENDING_STYLES = ("rising", "falling", "contained", "prolonged")
AVOIDANCES = (
    "不要播音腔", "不要播音主持腔", "不要过度磁性", "不要夸张表演",
    "不要幼态和甜腻", "不要儿童感", "不要尖锐", "不要刻意压低嗓音",
    "不要成熟大叔感", "不要明显沙哑",
)
_VISUAL_DESCRIPTION_TERMS = (
    "眼镜", "长发", "短发", "头发", "发型", "服饰", "衣服", "穿着",
    "西装", "裙子", "长袍", "甲胄", "盔甲", "制服", "古装", "妆容",
    "胡须", "配饰", "耳环", "帽子", "五官", "脸型", "姿态",
)


def _choice(value: Any, allowed: tuple[str, ...], default: str) -> str:
    normalized = str(value).strip() if value is not None else ""
    return normalized if normalized in allowed else default


def _visual_age(value: Any) -> int:
    match = re.search(r"\d+", str(value)) if value is not None else None
    if not match:
        return 25
    requested_age = int(match.group())
    return min(AGE_ANCHORS, key=lambda anchor: abs(anchor - requested_age))


def _character_identity(value: Any) -> str:
    """Keep a short, open-ended internal role concept without leaking markup."""
    normalized = re.sub(r"[\r\n\t]+", " ", str(value or "")).strip()
    normalized = re.sub(r"\s{2,}", " ", normalized)
    if any(term in normalized for term in _VISUAL_DESCRIPTION_TERMS):
        return "未明确角色"
    return normalized[:24] or "未明确角色"


def _short_text(value: Any, default: str, limit: int = 32) -> str:
    """Normalize compact internal visual/persona notes without leaking markup."""
    normalized = re.sub(r"[\r\n\t]+", " ", str(value or "")).strip()
    normalized = re.sub(r"\s{2,}", " ", normalized)
    return normalized[:limit] or default


def _persona_identity(value: Any) -> str:
    """Normalize one concrete role label without trailing lore or slogans."""
    normalized = re.sub(r"[\r\n\t]+", " ", str(value or "")).strip()
    normalized = re.sub(r"\s{2,}", " ", normalized)
    normalized = re.split(r"[，,；。]", normalized, maxsplit=1)[0].strip(
        " \"'“”‘’「」"
    )
    if any(term in normalized for term in _VISUAL_DESCRIPTION_TERMS):
        return "未明确角色"
    return normalized[:14] or "未明确角色"


def _persona_sentence(value: Any, default: str, limit: int = 28) -> str:
    """Keep complete compact clauses instead of cutting prose mid-sentence."""
    normalized = re.sub(r"[\r\n\t]+", " ", str(value or "")).strip()
    normalized = re.sub(r"\s{2,}", " ", normalized)
    sentence = re.split(r"[；。]", normalized, maxsplit=1)[0].strip("，,；。 ")
    if not sentence:
        return default
    if len(sentence) <= limit:
        return sentence

    selected: list[str] = []
    total_length = 0
    for clause in re.split(r"[，,]", sentence):
        clause = clause.strip("，,；。 ")
        if not clause:
            continue
        added_length = len(clause) + (1 if selected else 0)
        if selected and total_length + added_length > limit:
            break
        if not selected and len(clause) > limit:
            return clause[:limit].rstrip("，,；。 ")
        selected.append(clause)
        total_length += added_length
    return "，".join(selected) or default


def _free_style_choices(value: Any, default: tuple[str, ...]) -> tuple[str, ...]:
    """Keep two or three short Chinese internal descriptors."""
    if isinstance(value, str):
        candidates: Iterable[Any] = value.replace("、", ",").split(",")
    elif isinstance(value, (list, tuple)):
        candidates = value
    else:
        candidates = ()
    selected: list[str] = []
    for item in candidates:
        normalized = re.sub(r"\s+", "", str(item)).strip("，。；、")[:10]
        if normalized and re.search(r"[\u4e00-\u9fff]", normalized) and normalized not in selected:
            selected.append(normalized)
        if len(selected) == 3:
            break
    for fallback in default:
        if len(selected) >= 2:
            break
        if fallback not in selected:
            selected.append(fallback)
    return tuple(selected)


def _personality_choices(value: Any) -> tuple[str, ...]:
    """Keep concrete personality traits instead of clipped poetic sentences."""
    if isinstance(value, str):
        raw_candidates: Iterable[Any] = value.replace("、", ",").split(",")
    elif isinstance(value, (list, tuple)):
        raw_candidates = value
    else:
        raw_candidates = ()

    selected: list[str] = []
    for item in raw_candidates:
        for fragment in re.split(r"[，,、；。]", str(item)):
            normalized = re.sub(r"\s+", "", fragment).strip("，。；、")
            if not 2 <= len(normalized) <= 6:
                continue
            if not re.search(r"[\u4e00-\u9fff]", normalized):
                continue
            if re.search(
                r"如|似|仿佛|宛若|犹如|以.+为|星|剑|霜|天机|众生|虚妄|命运",
                normalized,
            ):
                continue
            if normalized not in selected:
                selected.append(normalized)
            if len(selected) == 3:
                return tuple(selected)
    for fallback in ("谨慎", "内省"):
        if len(selected) >= 2:
            break
        if fallback not in selected:
            selected.append(fallback)
    return tuple(selected)


def _list_choices(
    value: Any,
    allowed: tuple[str, ...],
    default: tuple[str, ...],
    limit: int,
) -> tuple[str, ...]:
    if isinstance(value, str):
        candidates: Iterable[Any] = value.replace("、", ",").split(",")
    elif isinstance(value, (list, tuple)):
        candidates = value
    else:
        candidates = ()
    selected: list[str] = []
    for item in candidates:
        normalized = str(item).strip()
        if normalized in allowed and normalized not in selected:
            selected.append(normalized)
        if len(selected) == limit:
            break
    return tuple(selected or default)


def _style_choices(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        candidates: Iterable[Any] = value.replace("、", ",").split(",")
    elif isinstance(value, (list, tuple)):
        candidates = value
    else:
        candidates = ()
    selected: list[str] = []
    for item in candidates:
        normalized = re.sub(r"\s+", "", str(item)).strip("，。；、")[:8]
        if (
            normalized
            and re.search(r"[\u4e00-\u9fff]", normalized)
            and not any(term in normalized for term in _VISUAL_DESCRIPTION_TERMS)
            and normalized not in selected
        ):
            selected.append(normalized)
        if len(selected) == 3:
            break
    for fallback in ("克制", "有控制力"):
        if len(selected) >= 2:
            break
        if fallback not in selected:
            selected.append(fallback)
    return tuple(selected)


@dataclass(frozen=True)
class VisualProfile:
    """Image-grounded observations used only as voice-design evidence."""

    visual_age_style: int
    gender_presentation: str
    facial_maturity: str
    visual_mass: str
    facial_contour: str
    body_silhouette: str
    styling_structure: str
    hair_style: str
    facial_hair: str
    clothing_style: str
    clothing_weight: str
    posture: str
    expression: str
    character_type: str
    visual_temperament: tuple[str, ...]

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "VisualProfile":
        return cls(
            visual_age_style=_visual_age(data.get("visual_age_style")),
            gender_presentation=_choice(
                data.get("gender_presentation"), VOICE_PRESENTATIONS, "androgynous"
            ),
            facial_maturity=_choice(
                data.get("facial_maturity"), FACIAL_MATURITIES, "young"
            ),
            visual_mass=_choice(data.get("visual_mass"), VISUAL_MASSES, "balanced"),
            facial_contour=_choice(
                data.get("facial_contour"), FACIAL_CONTOURS, "balanced"
            ),
            body_silhouette=_choice(
                data.get("body_silhouette"), BODY_SILHOUETTES, "balanced"
            ),
            styling_structure=_choice(
                data.get("styling_structure"), STYLING_STRUCTURES, "structured"
            ),
            hair_style=_short_text(data.get("hair_style"), "无明显特征"),
            facial_hair=_choice(
                data.get("facial_hair"), FACIAL_HAIR_LEVELS, "none"
            ),
            clothing_style=_short_text(data.get("clothing_style"), "日常简洁"),
            clothing_weight=_choice(
                data.get("clothing_weight"), CLOTHING_WEIGHTS, "medium"
            ),
            posture=_choice(data.get("posture"), POSTURES, "composed"),
            expression=_choice(data.get("expression"), EXPRESSIONS, "neutral"),
            character_type=_short_text(
                data.get("character_type"), "未明确角色", limit=24
            ),
            visual_temperament=_free_style_choices(
                data.get("visual_temperament"), ("克制", "内收")
            ),
        )

    def as_json_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["visual_temperament"] = list(self.visual_temperament)
        return result


@dataclass(frozen=True)
class VisualAnchorReview:
    """Focused second-pass review of identity-bearing visible anchors."""

    age_style: int
    gender_presentation: str
    facial_maturity: str
    visual_mass: str
    facial_contour: str
    body_silhouette: str
    styling_structure: str
    facial_hair: str
    confidence: str

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "VisualAnchorReview":
        return cls(
            age_style=_visual_age(data.get("age_style")),
            gender_presentation=_choice(
                data.get("gender_presentation"),
                VOICE_PRESENTATIONS,
                "androgynous",
            ),
            facial_maturity=_choice(
                data.get("facial_maturity"), FACIAL_MATURITIES, "young"
            ),
            visual_mass=_choice(
                data.get("visual_mass"), VISUAL_MASSES, "balanced"
            ),
            facial_contour=_choice(
                data.get("facial_contour"), FACIAL_CONTOURS, "balanced"
            ),
            body_silhouette=_choice(
                data.get("body_silhouette"), BODY_SILHOUETTES, "balanced"
            ),
            styling_structure=_choice(
                data.get("styling_structure"), STYLING_STRUCTURES, "structured"
            ),
            facial_hair=_choice(
                data.get("facial_hair"), FACIAL_HAIR_LEVELS, "none"
            ),
            confidence=_choice(
                data.get("confidence"), ANCHOR_CONFIDENCES, "medium"
            ),
        )

    def as_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def reconcile_visual_anchors(
    visual: VisualProfile, review: VisualAnchorReview
) -> VisualProfile:
    """Merge a focused review and repair obvious age/gender contradictions."""
    if review.confidence == "low":
        age_style = visual.visual_age_style
        gender_presentation = visual.gender_presentation
        facial_maturity = visual.facial_maturity
        visual_mass = visual.visual_mass
        facial_contour = visual.facial_contour
        body_silhouette = visual.body_silhouette
        styling_structure = visual.styling_structure
        facial_hair = visual.facial_hair
    elif review.confidence == "high":
        age_style = review.age_style
        gender_presentation = review.gender_presentation
        if (
            gender_presentation == "androgynous"
            and visual.gender_presentation != "androgynous"
        ):
            gender_presentation = visual.gender_presentation
        facial_maturity = review.facial_maturity
        visual_mass = review.visual_mass
        facial_contour = review.facial_contour
        body_silhouette = review.body_silhouette
        styling_structure = review.styling_structure
        facial_hair = review.facial_hair
    else:
        age_style = review.age_style
        gender_presentation = review.gender_presentation
        if (
            gender_presentation == "androgynous"
            and visual.gender_presentation != "androgynous"
        ):
            gender_presentation = visual.gender_presentation
        facial_maturity = max(
            (visual.facial_maturity, review.facial_maturity),
            key=FACIAL_MATURITIES.index,
        )
        visual_mass = max(
            (visual.visual_mass, review.visual_mass), key=VISUAL_MASSES.index
        )
        facial_contour = (
            review.facial_contour
            if review.facial_contour != "balanced"
            else visual.facial_contour
        )
        body_silhouette = (
            review.body_silhouette
            if review.body_silhouette != "balanced"
            else visual.body_silhouette
        )
        styling_structure = (
            review.styling_structure
            if review.styling_structure != "structured"
            else visual.styling_structure
        )
        facial_hair = max(
            (visual.facial_hair, review.facial_hair),
            key=FACIAL_HAIR_LEVELS.index,
        )

    if facial_maturity == "mature":
        age_style = max(age_style, 35)
    elif facial_maturity == "aged":
        age_style = max(age_style, 55)

    if facial_hair == "full":
        gender_presentation = "masculine"
        age_style = max(age_style, 35)
        if facial_maturity in {"mature", "aged"} and visual_mass in {
            "solid",
            "imposing",
        }:
            age_style = max(age_style, 38)

    # A trimmed beard can still be a strong maturity cue. The previous rule
    # only handled a full beard plus a heavy build, allowing lean, angular
    # adult men to remain in the young-adult voice band.
    mature_masculine_markers = sum(
        (
            facial_hair in {"light", "full"},
            facial_maturity in {"mature", "aged"},
            facial_contour in {"angular", "broad"},
            visual.expression in {"serious", "stern"},
        )
    )
    if (
        gender_presentation == "masculine"
        and age_style >= 30
        and mature_masculine_markers >= 2
    ):
        age_style = max(age_style, 35)
        facial_maturity = max(
            (facial_maturity, "mature"), key=FACIAL_MATURITIES.index
        )

    if facial_maturity == "mature" and visual_mass == "imposing":
        age_style = max(age_style, 35)

    return replace(
        visual,
        visual_age_style=_visual_age(age_style),
        gender_presentation=gender_presentation,
        facial_maturity=facial_maturity,
        visual_mass=visual_mass,
        facial_contour=facial_contour,
        body_silhouette=body_silhouette,
        styling_structure=styling_structure,
        facial_hair=facial_hair,
    )


@dataclass(frozen=True)
class PersonaProfile:
    """Open-ended fictional role core, never a claim about the real person."""

    gender: str
    age_style: int
    identity: str
    core_personality: tuple[str, ...]
    archetype: str
    moral_tendency: str
    motivation: str
    social_style: str
    emotional_pattern: str

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "PersonaProfile":
        return cls(
            gender=_choice(data.get("gender"), VOICE_PRESENTATIONS, "androgynous"),
            age_style=_visual_age(data.get("age_style")),
            identity=_persona_identity(data.get("identity")),
            core_personality=_personality_choices(data.get("core_personality")),
            archetype=_persona_identity(data.get("archetype") or "观察型"),
            moral_tendency=_persona_sentence(
                data.get("moral_tendency"), "务实中立", limit=24
            ),
            motivation=_persona_sentence(
                data.get("motivation"), "维持自身秩序", limit=28
            ),
            social_style=_persona_sentence(
                data.get("social_style"), "保持礼貌距离", limit=28
            ),
            emotional_pattern=_persona_sentence(
                data.get("emotional_pattern"), "情绪外显有限", limit=28
            ),
        )

    def generic_dimension_count(self) -> int:
        generic_pairs = (
            (self.identity, "未明确角色"),
            (self.archetype, "观察型"),
            (self.moral_tendency, "务实中立"),
            (self.motivation, "维持自身秩序"),
            (self.social_style, "保持礼貌距离"),
            (self.emotional_pattern, "情绪外显有限"),
        )
        return sum(actual == generic for actual, generic in generic_pairs) + int(
            self.core_personality == ("谨慎", "内省")
        )

    def quality_issue_count(self) -> int:
        """Count generic, visual-leaking, poetic or visibly incomplete content."""
        combined = " ".join(
            (
                self.identity,
                "、".join(self.core_personality),
                self.archetype,
                self.moral_tendency,
                self.motivation,
                self.social_style,
                self.emotional_pattern,
            )
        )
        issues = self.generic_dimension_count()
        issues += int(any(term in combined for term in _VISUAL_DESCRIPTION_TERMS))
        issues += int(
            bool(
                re.search(
                    r"仿佛|宛若|犹如|为骨|为魂|如霜|似剑|"
                    r"星陨|星穹|星痕|星图|星核|蚀痕|天机|心灯|剑道|"
                    r"众生|虚妄|命运之|[·•]",
                    combined,
                )
            )
        )
        incomplete_endings = ("不", "为", "在", "与", "而", "于", "的", "以", "只")
        for value in (
            self.moral_tendency,
            self.motivation,
            self.social_style,
            self.emotional_pattern,
        ):
            issues += int(value.endswith(incomplete_endings))
        return issues

    def as_json_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["core_personality"] = list(self.core_personality)
        return result


@dataclass(frozen=True)
class SpeakingBehavior:
    """Stable, character-specific habits for how sentences are delivered."""

    speech_rate: str
    pause_pattern: str
    rhythm_style: str
    articulation_style: str
    emotional_reactivity: str
    restraint_style: str
    sentence_energy: str
    ending_style: str
    vocal_distance: str

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "SpeakingBehavior":
        return cls(
            speech_rate=_choice(data.get("speech_rate"), BASE_SPEECH_RATES, "medium"),
            pause_pattern=_choice(
                data.get("pause_pattern"), PAUSE_PATTERNS, "regular"
            ),
            rhythm_style=_choice(data.get("rhythm_style"), RHYTHM_STYLES, "even"),
            articulation_style=_choice(
                data.get("articulation_style"), BEHAVIOR_ARTICULATIONS, "natural"
            ),
            emotional_reactivity=_choice(
                data.get("emotional_reactivity"),
                EMOTIONAL_REACTIVITIES,
                "medium",
            ),
            restraint_style=_choice(
                data.get("restraint_style"), RESTRAINT_STYLES, "balanced"
            ),
            sentence_energy=_choice(
                data.get("sentence_energy"), SENTENCE_ENERGIES, "steady"
            ),
            ending_style=_choice(
                data.get("ending_style"), ENDING_STYLES, "contained"
            ),
            vocal_distance=_choice(
                data.get("vocal_distance"), VOCAL_DISTANCES, "natural"
            ),
        )

    def neutral_dimension_count(self) -> int:
        neutral_pairs = (
            (self.speech_rate, "medium"),
            (self.pause_pattern, "regular"),
            (self.rhythm_style, "even"),
            (self.articulation_style, "natural"),
            (self.emotional_reactivity, "medium"),
            (self.restraint_style, "balanced"),
            (self.sentence_energy, "steady"),
            (self.vocal_distance, "natural"),
        )
        return sum(actual == neutral for actual, neutral in neutral_pairs)

    def as_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BaseVoiceProfile:
    visual_age_style: int
    voice_presentation: str
    pitch_center: str
    pitch_range: str
    vocal_weight: str
    brightness: str
    resonance_position: str
    resonance_depth: str
    breathiness: str
    nasality: str
    texture: str
    roughness: str
    onset: str
    articulation: str
    vocal_tension: str
    warmth: str
    vocal_distance: str
    maturity: str
    vocal_maturity: str
    character_identity: str
    speech_rate: str
    dynamic_range: str
    pause_pattern: str
    rhythm_style: str
    ending_style: str
    overall_voice_style: tuple[str, ...]
    avoidances: tuple[str, ...]

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "BaseVoiceProfile":
        return cls(
            visual_age_style=_visual_age(data.get("visual_age_style")),
            voice_presentation=_choice(data.get("voice_presentation"), VOICE_PRESENTATIONS, "androgynous"),
            pitch_center=_choice(data.get("pitch_center"), PITCH_CENTERS, "medium"),
            pitch_range=_choice(data.get("pitch_range"), PITCH_RANGES, "moderate"),
            vocal_weight=_choice(data.get("vocal_weight"), VOCAL_WEIGHTS, "medium"),
            brightness=_choice(data.get("brightness"), BRIGHTNESSES, "neutral"),
            resonance_position=_choice(data.get("resonance_position"), RESONANCE_POSITIONS, "mixed"),
            resonance_depth=_choice(data.get("resonance_depth"), RESONANCE_DEPTHS, "medium"),
            breathiness=_choice(data.get("breathiness"), BREATHINESS_LEVELS, "slight"),
            nasality=_choice(data.get("nasality"), NASALITY_LEVELS, "none"),
            texture=_choice(data.get("texture"), TEXTURES, "clean"),
            roughness=_choice(data.get("roughness"), ROUGHNESS_LEVELS, "none"),
            onset=_choice(data.get("onset"), ONSETS, "balanced"),
            articulation=_choice(data.get("articulation"), ARTICULATIONS, "natural"),
            vocal_tension=_choice(data.get("vocal_tension"), VOCAL_TENSIONS, "neutral"),
            warmth=_choice(data.get("warmth"), WARMTH_LEVELS, "neutral"),
            vocal_distance=_choice(data.get("vocal_distance"), VOCAL_DISTANCES, "natural"),
            maturity=_choice(data.get("maturity"), MATURITY_LEVELS, "young"),
            vocal_maturity=_choice(
                data.get("vocal_maturity"), VOCAL_MATURITIES, "young_adult"
            ),
            character_identity=_character_identity(data.get("character_identity")),
            speech_rate=_choice(
                data.get("speech_rate"), BASE_SPEECH_RATES, "medium"
            ),
            dynamic_range=_choice(
                data.get("dynamic_range"), DYNAMIC_RANGES, "moderate"
            ),
            pause_pattern=_choice(
                data.get("pause_pattern"), PAUSE_PATTERNS, "regular"
            ),
            rhythm_style=_choice(
                data.get("rhythm_style"), RHYTHM_STYLES, "even"
            ),
            ending_style=_choice(
                data.get("ending_style"), ENDING_STYLES, "contained"
            ),
            overall_voice_style=_style_choices(data.get("overall_voice_style")),
            avoidances=_list_choices(data.get("avoidances"), AVOIDANCES, (), 2),
        )

    def neutral_dimension_count(self) -> int:
        """Count generic choices that can collapse different characters together."""
        neutral_pairs = (
            (self.pitch_center, "medium"),
            (self.pitch_range, "moderate"),
            (self.vocal_weight, "medium"),
            (self.brightness, "neutral"),
            (self.resonance_depth, "medium"),
            (self.breathiness, "medium"),
            (self.dynamic_range, "moderate"),
            (self.onset, "balanced"),
            (self.articulation, "natural"),
            (self.vocal_tension, "neutral"),
            (self.warmth, "neutral"),
            (self.vocal_distance, "natural"),
            (self.vocal_maturity, "young_adult"),
            (self.pause_pattern, "regular"),
            (self.rhythm_style, "even"),
        )
        return sum(actual == neutral for actual, neutral in neutral_pairs)

    def as_json_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["overall_voice_style"] = list(self.overall_voice_style)
        result["avoidances"] = list(self.avoidances)
        return result


def repair_acoustic_coherence(
    profile: BaseVoiceProfile,
    *,
    locked_fields: Iterable[str] = (),
) -> BaseVoiceProfile:
    """Repair combinations that tend to collapse or confuse VoiceDesign.

    The model may independently choose individually valid enum values that form
    a contradictory instruction, such as a heavy voice with shallow head
    resonance or a soft onset with high tension. Explicit user overrides can be
    locked so this normalization never silently cancels them.
    """
    locked = set(locked_fields)
    updates: dict[str, str] = {}

    def current(field_name: str) -> str:
        return updates.get(field_name, getattr(profile, field_name))

    def set_if_unlocked(field_name: str, value: str) -> None:
        if field_name not in locked:
            updates[field_name] = value

    weight = current("vocal_weight")
    position = current("resonance_position")
    depth = current("resonance_depth")
    if weight in {"full", "heavy"}:
        if position == "head":
            set_if_unlocked("resonance_position", "mixed")
        if depth == "shallow":
            set_if_unlocked("resonance_depth", "medium")
    elif weight in {"very_light", "light"}:
        if position == "chest":
            set_if_unlocked("resonance_position", "mixed")
        if depth == "deep":
            set_if_unlocked("resonance_depth", "medium")

    position = current("resonance_position")
    depth = current("resonance_depth")
    if position in {"head", "front"} and depth == "deep":
        set_if_unlocked("resonance_depth", "medium")
    elif position == "chest" and depth == "shallow":
        set_if_unlocked("resonance_depth", "medium")

    roughness = current("roughness")
    texture = current("texture")
    if roughness in {"medium", "strong"} and texture in {
        "clean",
        "silky",
        "airy",
        "crisp",
    }:
        set_if_unlocked("texture", "grainy")
    elif roughness == "none" and texture in {"grainy", "husky"}:
        set_if_unlocked("roughness", "slight")

    onset = current("onset")
    tension = current("vocal_tension")
    if onset == "firm" and tension == "relaxed":
        set_if_unlocked("vocal_tension", "neutral")
    elif onset == "soft" and tension == "tense":
        set_if_unlocked("vocal_tension", "neutral")
    if current("vocal_distance") == "projected" and current("onset") == "soft":
        set_if_unlocked("onset", "balanced")
    if current("breathiness") == "strong" and current("onset") == "firm":
        set_if_unlocked("onset", "balanced")

    return replace(profile, **updates) if updates else profile


_VISUAL_MASS_SCORES = {
    "delicate": -4,
    "light": -2,
    "balanced": 0,
    "solid": 2,
    "imposing": 4,
}
_CLOTHING_WEIGHT_SCORES = {"light": -1, "medium": 0, "heavy": 1}
_POSTURE_SCORES = {
    "relaxed": -1,
    "composed": 0,
    "upright": 1,
    "dynamic": 2,
    "commanding": 3,
}
_EXPRESSION_SCORES = {
    "gentle": -1,
    "neutral": 0,
    "serious": 1,
    "stern": 2,
    "lively": 1,
}


def visual_structure_score(visual: VisualProfile) -> int:
    """Combine general visible structure cues without using role-name templates."""
    return (
        _VISUAL_MASS_SCORES[visual.visual_mass]
        + _CLOTHING_WEIGHT_SCORES[visual.clothing_weight]
        + _POSTURE_SCORES[visual.posture]
        + _EXPRESSION_SCORES[visual.expression]
    )


def align_voice_to_visual_structure(
    visual: VisualProfile,
    profile: BaseVoiceProfile,
) -> BaseVoiceProfile:
    """Map independent visible structure axes into audible voice structure.

    This is a consistency guard, not a claim about a real person's voice. It
    separates actual body outline, facial line and styling rigidity so that a
    wide robe or armor does not make every character full, dark and chest-led.
    """
    score = visual_structure_score(visual)
    updates: dict[str, Any] = {}

    # Keep age-bearing facial evidence audible. This guard changes concrete
    # acoustic controls instead of relying on an abstract "mature" label that
    # VoiceDesign may underweight.
    if visual.visual_age_style >= MATURE_ADULT_AGE:
        updates["maturity"] = "mature"
        if profile.vocal_maturity in {"youthful", "young_adult"}:
            updates["vocal_maturity"] = (
                "mature"
                if visual.visual_age_style >= 35
                else "mature_young"
            )
    if (
        visual.gender_presentation == "masculine"
        and visual.facial_hair in {"light", "full"}
        and visual.facial_maturity in {"mature", "aged"}
    ):
        updates["vocal_maturity"] = (
            "seasoned" if visual.visual_age_style >= 55 else "mature"
        )
        if profile.pitch_center in {"medium_high", "high"}:
            updates["pitch_center"] = (
                "medium_low" if visual.facial_hair == "full" else "medium"
            )
        if profile.vocal_weight in {"very_light", "light"}:
            updates["vocal_weight"] = "medium"
        if profile.brightness in {"bright", "very_bright"}:
            updates["brightness"] = "neutral"
        if profile.resonance_position == "head":
            updates["resonance_position"] = "mixed"
        if profile.resonance_depth == "shallow":
            updates["resonance_depth"] = "medium"
        updates["avoidances"] = tuple(
            item for item in profile.avoidances if item != "不要成熟大叔感"
        )

    if score >= 4:
        if profile.vocal_weight in {"very_light", "light"}:
            updates["vocal_weight"] = (
                "full" if visual.body_silhouette == "broad" else "medium"
            )
        if profile.resonance_depth == "shallow":
            updates["resonance_depth"] = "medium"
        if profile.resonance_position == "head":
            updates["resonance_position"] = "mixed"
        if visual.posture in {"dynamic", "commanding"} or visual.expression == "stern":
            if profile.vocal_distance == "intimate":
                updates["vocal_distance"] = "projected"
        elif profile.vocal_distance == "intimate":
            updates["vocal_distance"] = "natural"
        if profile.onset == "soft":
            updates["onset"] = "balanced"
        if profile.vocal_tension == "relaxed":
            updates["vocal_tension"] = (
                "tense" if visual.posture == "commanding" else "neutral"
            )
    elif score <= -2 and visual.visual_mass in {"delicate", "light"}:
        if profile.vocal_weight in {"full", "heavy"}:
            updates["vocal_weight"] = (
                "light" if visual.body_silhouette == "slender" else "medium"
            )
        if profile.resonance_depth == "deep":
            updates["resonance_depth"] = "medium"
        if profile.resonance_position == "chest":
            updates["resonance_position"] = "mixed"
        if profile.vocal_distance == "projected" and visual.posture in {
            "relaxed",
            "composed",
        }:
            updates["vocal_distance"] = "natural"
        if profile.onset == "firm" and visual.posture in {"relaxed", "composed"}:
            updates["onset"] = "balanced"
        if profile.vocal_tension == "tense" and visual.expression in {
            "gentle",
            "neutral",
        }:
            updates["vocal_tension"] = "neutral"

    # Body silhouette controls the size and depth of the audible voice body.
    # It is intentionally evaluated after the broad mass score so bulky
    # clothing cannot hide a visibly slender frame.
    current_weight = updates.get("vocal_weight", profile.vocal_weight)
    current_position = updates.get(
        "resonance_position", profile.resonance_position
    )
    current_depth = updates.get("resonance_depth", profile.resonance_depth)
    if visual.body_silhouette == "slender":
        if current_weight in {"full", "heavy"}:
            updates["vocal_weight"] = (
                "light" if visual.visual_mass in {"delicate", "light"} else "medium"
            )
        if current_position == "chest":
            updates["resonance_position"] = "mixed"
        if current_depth == "deep":
            updates["resonance_depth"] = "medium"
    elif visual.body_silhouette == "broad":
        if current_weight in {"very_light", "light"}:
            updates["vocal_weight"] = "full"
        if current_position in {"head", "front"}:
            updates["resonance_position"] = (
                "chest" if visual.visual_mass in {"solid", "imposing"} else "mixed"
            )
        if current_depth == "shallow":
            updates["resonance_depth"] = (
                "deep" if visual.visual_mass in {"solid", "imposing"} else "medium"
            )

    # Facial line and styling rigidity affect surface, attack and diction.
    # These mappings create audible structure rather than leaking visual words
    # into the final VoiceDesign instruction.
    current_texture = updates.get("texture", profile.texture)
    if visual.styling_structure == "armored":
        if visual.facial_contour == "angular":
            if current_texture in {"clean", "silky", "airy"}:
                updates["texture"] = "metallic"
            if profile.roughness == "none":
                updates["roughness"] = "slight"
            updates["onset"] = "firm"
            updates["articulation"] = "crisp"
            if profile.vocal_tension == "relaxed":
                updates["vocal_tension"] = "neutral"
        elif visual.facial_contour == "broad":
            if current_texture in {"clean", "silky", "airy", "crisp"}:
                updates["texture"] = "grainy"
            if profile.roughness == "none":
                updates["roughness"] = "slight"
    elif visual.styling_structure == "fluid" and visual.facial_contour == "soft":
        if current_texture in {"metallic", "grainy"} and profile.roughness in {
            "none",
            "slight",
        }:
            updates["texture"] = "silky"
        if profile.onset == "firm":
            updates["onset"] = "balanced"
        if profile.articulation == "firm":
            updates["articulation"] = "natural"
    elif visual.facial_contour == "angular":
        if current_texture in {"silky", "airy"}:
            updates["texture"] = "crisp"
        if profile.articulation == "soft":
            updates["articulation"] = "crisp"

    aligned = replace(profile, **updates) if updates else profile
    return repair_acoustic_coherence(aligned)


@dataclass(frozen=True)
class VoiceDesignReference:
    """One prior batch result; legacy registries may lack upper layers."""

    voice_fingerprint: BaseVoiceProfile
    persona_profile: PersonaProfile | None = None
    speaking_behavior: SpeakingBehavior | None = None


def _text_similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, left.strip(), right.strip()).ratio()


def persona_similarity(candidate: PersonaProfile, reference: PersonaProfile) -> float:
    """Compare role cores, emphasizing identity and behavioral motivation."""
    weighted_scores = (
        (0.25, 1.0 if candidate.gender == reference.gender else 0.0),
        (
            0.5,
            1.0 - min(abs(candidate.age_style - reference.age_style), 30) / 30,
        ),
        (2.0, _text_similarity(candidate.identity, reference.identity)),
        (
            2.0,
            _text_similarity(
                "、".join(candidate.core_personality),
                "、".join(reference.core_personality),
            ),
        ),
        (1.5, _text_similarity(candidate.archetype, reference.archetype)),
        (1.0, _text_similarity(candidate.moral_tendency, reference.moral_tendency)),
        (1.5, _text_similarity(candidate.motivation, reference.motivation)),
        (1.5, _text_similarity(candidate.social_style, reference.social_style)),
        (
            1.5,
            _text_similarity(
                candidate.emotional_pattern, reference.emotional_pattern
            ),
        ),
    )
    total_weight = sum(weight for weight, _ in weighted_scores)
    return sum(weight * score for weight, score in weighted_scores) / total_weight


def speaking_behavior_similarity(
    candidate: SpeakingBehavior, reference: SpeakingBehavior
) -> float:
    """Compare stable speaking habits, not temporary requested emotion."""
    weights = {
        "speech_rate": 1.0,
        "pause_pattern": 1.5,
        "rhythm_style": 1.5,
        "articulation_style": 1.25,
        "emotional_reactivity": 1.0,
        "restraint_style": 1.25,
        "sentence_energy": 1.25,
        "ending_style": 1.5,
        "vocal_distance": 0.75,
    }
    matched = sum(
        weight
        for field_name, weight in weights.items()
        if getattr(candidate, field_name) == getattr(reference, field_name)
    )
    return matched / sum(weights.values())


HIGH_WEIGHT_ACOUSTIC_FIELDS = (
    "vocal_maturity",
    "pitch_center",
    "vocal_weight",
    "brightness",
    "resonance_position",
    "resonance_depth",
    "onset",
    "articulation",
)
TIMBRE_IDENTITY_FIELDS = (
    "vocal_maturity",
    "pitch_center",
    "vocal_weight",
    "brightness",
    "resonance_position",
    "resonance_depth",
    "texture",
)
# These stable, audible fields form the scalable timbre space. Delivery-only
# fields such as rate, pauses, emotion and distance deliberately do not count.
PRIMARY_TIMBRE_FIELDS = (
    "vocal_maturity",
    "pitch_center",
    "pitch_range",
    "vocal_weight",
    "brightness",
    "warmth",
    "resonance_position",
    "resonance_depth",
    "breathiness",
    "nasality",
    "texture",
    "roughness",
    "onset",
    "vocal_tension",
    "articulation",
)
TIMBRE_SKELETON_FIELDS = (
    "vocal_maturity",
    "pitch_center",
    "vocal_weight",
    "brightness",
    "resonance_position",
    "resonance_depth",
    "texture",
    "roughness",
)
BODY_RESONANCE_FIELDS = (
    "vocal_weight",
    "resonance_position",
    "resonance_depth",
)
TONAL_SURFACE_FIELDS = ("brightness", "warmth", "texture", "roughness")
PRIMARY_TIMBRE_GROUPS = (
    ("vocal_maturity", "pitch_center", "pitch_range"),
    BODY_RESONANCE_FIELDS,
    TONAL_SURFACE_FIELDS,
    ("breathiness", "nasality", "onset", "vocal_tension", "articulation"),
)
CORE_ACOUSTIC_FIELDS = HIGH_WEIGHT_ACOUSTIC_FIELDS + (
    "pitch_range",
    "roughness",
    "vocal_tension",
    "warmth",
    "breathiness",
    "nasality",
)
AUXILIARY_ACOUSTIC_FIELDS = ("breathiness", "nasality")
ACOUSTIC_FIELD_WEIGHTS = {
    **{field: 2.0 for field in HIGH_WEIGHT_ACOUSTIC_FIELDS},
    "pitch_range": 1.0,
    "roughness": 1.0,
    "vocal_tension": 1.0,
    "breathiness": 0.25,
    "nasality": 0.25,
    "texture": 2.0,
}
_STRONG_DIFFERENCE_ORDERS: dict[str, tuple[str, ...]] = {
    "vocal_maturity": VOCAL_MATURITIES,
    "pitch_center": PITCH_CENTERS,
    "pitch_range": PITCH_RANGES,
    "vocal_weight": VOCAL_WEIGHTS,
    "brightness": BRIGHTNESSES,
    "resonance_position": RESONANCE_POSITIONS,
    "resonance_depth": RESONANCE_DEPTHS,
    "roughness": ROUGHNESS_LEVELS,
    "breathiness": BREATHINESS_LEVELS,
    "nasality": NASALITY_LEVELS,
    "onset": ONSETS,
    "vocal_tension": VOCAL_TENSIONS,
    "articulation": ARTICULATIONS,
    "warmth": WARMTH_LEVELS,
}


def acoustic_contrast(
    candidate: BaseVoiceProfile, reference: BaseVoiceProfile
) -> tuple[int, int]:
    """Return differing core fields and differences spanning at least two levels."""
    differences = 0
    strong_differences = 0
    for field_name in CORE_ACOUSTIC_FIELDS:
        candidate_value = getattr(candidate, field_name)
        reference_value = getattr(reference, field_name)
        if candidate_value == reference_value:
            continue
        differences += 1
        order = _STRONG_DIFFERENCE_ORDERS[field_name]
        if abs(order.index(candidate_value) - order.index(reference_value)) >= 2:
            strong_differences += 1
    return differences, strong_differences


def weighted_acoustic_distance(
    candidate: BaseVoiceProfile, reference: BaseVoiceProfile
) -> float:
    """Score identity-bearing fields highly and auxiliary colorations lightly."""
    return sum(
        weight
        for field_name, weight in ACOUSTIC_FIELD_WEIGHTS.items()
        if getattr(candidate, field_name) != getattr(reference, field_name)
    )


def timbre_identity_contrast(
    candidate: BaseVoiceProfile, reference: BaseVoiceProfile
) -> int:
    """Count differences in fields that most directly shape perceived timbre."""
    return sum(
        getattr(candidate, field_name) != getattr(reference, field_name)
        for field_name in TIMBRE_IDENTITY_FIELDS
    )


def primary_timbre_contrast(
    candidate: BaseVoiceProfile, reference: BaseVoiceProfile
) -> tuple[int, int, int]:
    """Return real timbre differences, strong differences and changed groups."""
    differences = 0
    strong_differences = 0
    for field_name in PRIMARY_TIMBRE_FIELDS:
        candidate_value = getattr(candidate, field_name)
        reference_value = getattr(reference, field_name)
        if candidate_value == reference_value:
            continue
        differences += 1
        order = _STRONG_DIFFERENCE_ORDERS.get(field_name)
        if order is None:
            # Texture is categorical: clean versus metallic/grainy is already
            # an audible surface identity change rather than a minor step.
            strong_differences += 1
        elif abs(order.index(candidate_value) - order.index(reference_value)) >= 2:
            strong_differences += 1

    changed_groups = sum(
        any(
            getattr(candidate, field_name) != getattr(reference, field_name)
            for field_name in group
        )
        for group in PRIMARY_TIMBRE_GROUPS
    )
    return differences, strong_differences, changed_groups


def _field_group_contrast(
    candidate: BaseVoiceProfile,
    reference: BaseVoiceProfile,
    fields: Iterable[str],
) -> int:
    return sum(
        getattr(candidate, field_name) != getattr(reference, field_name)
        for field_name in fields
    )


def _vocal_maturity_matches_visual_age(profile: BaseVoiceProfile) -> bool:
    age = profile.visual_age_style
    allowed = (
        {"youthful", "young_adult"}
        if age <= 18
        else {"youthful", "young_adult", "mature_young"}
        if age <= 25
        else {"young_adult", "mature_young"}
        if age < MATURE_ADULT_AGE
        else {"mature_young", "mature"}
        if age <= 50
        else {"mature", "seasoned"}
    )
    return profile.vocal_maturity in allowed


def voice_fingerprint_similarity(
    candidate: BaseVoiceProfile, reference: BaseVoiceProfile
) -> float:
    maximum_distance = sum(ACOUSTIC_FIELD_WEIGHTS.values())
    return 1.0 - weighted_acoustic_distance(candidate, reference) / maximum_distance


def layered_design_is_too_similar(
    persona_profile: PersonaProfile,
    speaking_behavior: SpeakingBehavior,
    voice_fingerprint: BaseVoiceProfile,
    reference: VoiceDesignReference,
    *,
    persona_threshold: float = 0.72,
    behavior_threshold: float = 0.67,
    voice_threshold: float = 0.62,
) -> bool:
    """Require simultaneous similarity across persona, behavior and acoustics."""
    if reference.persona_profile is None or reference.speaking_behavior is None:
        return False
    return (
        persona_similarity(persona_profile, reference.persona_profile)
        >= persona_threshold
        and speaking_behavior_similarity(
            speaking_behavior, reference.speaking_behavior
        )
        >= behavior_threshold
        and voice_fingerprint_similarity(
            voice_fingerprint, reference.voice_fingerprint
        )
        >= voice_threshold
    )


def layered_design_similarity(
    persona_profile: PersonaProfile,
    speaking_behavior: SpeakingBehavior,
    voice_fingerprint: BaseVoiceProfile,
    reference: VoiceDesignReference,
) -> float:
    """Return a comparable score for selecting the closest prior design."""
    voice_score = voice_fingerprint_similarity(
        voice_fingerprint, reference.voice_fingerprint
    )
    if reference.persona_profile is None or reference.speaking_behavior is None:
        return voice_score
    return (
        0.4 * persona_similarity(persona_profile, reference.persona_profile)
        + 0.35
        * speaking_behavior_similarity(
            speaking_behavior, reference.speaking_behavior
        )
        + 0.25 * voice_score
    )


def similar_demographic_core_contrast(
    candidate: BaseVoiceProfile, reference: BaseVoiceProfile
) -> int:
    """Require strong core separation for visually similar demographic anchors.

    This rule is global: it applies to any same-presentation pair with nearby
    visual ages, instead of special-casing young feminine voices.
    """
    if not (
        candidate.voice_presentation == reference.voice_presentation
        and abs(candidate.visual_age_style - reference.visual_age_style) <= 10
    ):
        return len(HIGH_WEIGHT_ACOUSTIC_FIELDS)
    return sum(
        getattr(candidate, field_name) != getattr(reference, field_name)
        for field_name in HIGH_WEIGHT_ACOUSTIC_FIELDS
    )


def young_feminine_core_contrast(
    candidate: BaseVoiceProfile, reference: BaseVoiceProfile
) -> int:
    """Backward-compatible alias for older callers."""
    return similar_demographic_core_contrast(candidate, reference)


def _normalized_order_distance(
    field_name: str, left: str, right: str
) -> float:
    order = _STRONG_DIFFERENCE_ORDERS[field_name]
    if left == right:
        return 0.0
    return abs(order.index(left) - order.index(right)) / max(len(order) - 1, 1)


def macro_voice_contrast(
    candidate: BaseVoiceProfile, reference: BaseVoiceProfile
) -> tuple[int, float]:
    """Compare perceptual voice silhouette instead of just counting fields.

    Returns:
      - number of macro axes with a clearly audible structural difference
      - mean normalized macro distance

    The grouped axes are register/maturity, body+resonance, tonal surface,
    attack/articulation, and prosodic span. This prevents five tiny enum changes
    from being treated as a genuinely different voice.
    """
    texture_distance = 0.0 if candidate.texture == reference.texture else 0.5
    groups = (
        (
            _normalized_order_distance(
                "vocal_maturity",
                candidate.vocal_maturity,
                reference.vocal_maturity,
            )
            + _normalized_order_distance(
                "pitch_center", candidate.pitch_center, reference.pitch_center
            )
        )
        / 2.0,
        (
            _normalized_order_distance(
                "vocal_weight", candidate.vocal_weight, reference.vocal_weight
            )
            + _normalized_order_distance(
                "resonance_position",
                candidate.resonance_position,
                reference.resonance_position,
            )
            + _normalized_order_distance(
                "resonance_depth",
                candidate.resonance_depth,
                reference.resonance_depth,
            )
        )
        / 3.0,
        (
            _normalized_order_distance(
                "brightness", candidate.brightness, reference.brightness
            )
            + _normalized_order_distance(
                "roughness", candidate.roughness, reference.roughness
            )
            + texture_distance
        )
        / 3.0,
        (
            _normalized_order_distance("onset", candidate.onset, reference.onset)
            + _normalized_order_distance(
                "vocal_tension",
                candidate.vocal_tension,
                reference.vocal_tension,
            )
            + _normalized_order_distance(
                "articulation",
                candidate.articulation,
                reference.articulation,
            )
        )
        / 3.0,
        _normalized_order_distance(
            "pitch_range", candidate.pitch_range, reference.pitch_range
        ),
    )
    clearly_different_axes = sum(score >= 0.34 for score in groups)
    return clearly_different_axes, sum(groups) / len(groups)


def profile_meets_contrast(
    candidate: BaseVoiceProfile,
    references: Iterable[BaseVoiceProfile],
    *,
    minimum_differences: int = 5,
    minimum_strong_differences: int = 3,
    minimum_weighted_distance: float = 8.0,
    minimum_timbre_differences: int = 2,
    minimum_primary_timbre_differences: int = 4,
    minimum_primary_timbre_strong_differences: int = 1,
    minimum_primary_timbre_groups: int = 2,
    minimum_timbre_skeleton_differences: int = 2,
    similar_demographic_minimum_core_differences: int = 4,
    minimum_macro_axes: int = 2,
    minimum_macro_distance: float = 0.22,
) -> bool:
    """Require field-level and macro-perceptual separation from every reference."""
    if not _vocal_maturity_matches_visual_age(candidate):
        return False
    for reference in references:
        differences, strong_differences = acoustic_contrast(candidate, reference)
        macro_axes, macro_distance = macro_voice_contrast(candidate, reference)
        (
            primary_differences,
            primary_strong_differences,
            primary_groups,
        ) = primary_timbre_contrast(candidate, reference)
        body_resonance_differences = _field_group_contrast(
            candidate, reference, BODY_RESONANCE_FIELDS
        )
        tonal_surface_differences = _field_group_contrast(
            candidate, reference, TONAL_SURFACE_FIELDS
        )
        timbre_skeleton_differences = _field_group_contrast(
            candidate, reference, TIMBRE_SKELETON_FIELDS
        )
        if (
            differences < minimum_differences
            or strong_differences < minimum_strong_differences
            or weighted_acoustic_distance(candidate, reference)
            < minimum_weighted_distance
            or timbre_identity_contrast(candidate, reference)
            < minimum_timbre_differences
            or primary_differences < minimum_primary_timbre_differences
            or primary_strong_differences
            < minimum_primary_timbre_strong_differences
            or primary_groups < minimum_primary_timbre_groups
            or timbre_skeleton_differences
            < minimum_timbre_skeleton_differences
            or similar_demographic_core_contrast(candidate, reference)
            < similar_demographic_minimum_core_differences
            # If the audible body/resonance is identical, require at least two
            # surface changes. One adjacent brightness word is not enough to
            # distinguish otherwise identical voices (the result-3/5 case).
            or (body_resonance_differences < 1 and tonal_surface_differences < 2)
            or macro_axes < minimum_macro_axes
            or macro_distance < minimum_macro_distance
        ):
            return False
    return True


def allocate_contrasting_profile(
    seed: BaseVoiceProfile,
    references: Iterable[BaseVoiceProfile],
    *,
    minimum_differences: int = 5,
    minimum_strong_differences: int = 3,
    minimum_weighted_distance: float = 8.0,
    minimum_timbre_differences: int = 2,
    minimum_primary_timbre_differences: int = 4,
    minimum_primary_timbre_strong_differences: int = 1,
    minimum_primary_timbre_groups: int = 2,
    minimum_timbre_skeleton_differences: int = 2,
    similar_demographic_minimum_core_differences: int = 4,
    minimum_macro_axes: int = 2,
    minimum_macro_distance: float = 0.22,
    locked_fields: Iterable[str] = (),
    visual_profile: VisualProfile | None = None,
) -> BaseVoiceProfile | None:
    """Find the closest coherent core signature that passes batch contrast."""
    reference_list = list(references)
    if profile_meets_contrast(
        seed,
        reference_list,
        minimum_differences=minimum_differences,
        minimum_strong_differences=minimum_strong_differences,
        minimum_weighted_distance=minimum_weighted_distance,
        minimum_timbre_differences=minimum_timbre_differences,
        minimum_primary_timbre_differences=minimum_primary_timbre_differences,
        minimum_primary_timbre_strong_differences=(
            minimum_primary_timbre_strong_differences
        ),
        minimum_primary_timbre_groups=minimum_primary_timbre_groups,
        minimum_timbre_skeleton_differences=(
            minimum_timbre_skeleton_differences
        ),
        similar_demographic_minimum_core_differences=(
            similar_demographic_minimum_core_differences
        ),
        minimum_macro_axes=minimum_macro_axes,
        minimum_macro_distance=minimum_macro_distance,
    ):
        return seed

    locked = set(locked_fields) & set(CORE_ACOUSTIC_FIELDS)
    reference_values = [
        tuple(getattr(reference, field) for field in CORE_ACOUSTIC_FIELDS)
        for reference in reference_list
    ]
    seed_values = tuple(getattr(seed, field) for field in CORE_ACOUSTIC_FIELDS)
    usage_counts = {
        field: {
            value: sum(getattr(reference, field) == value for reference in reference_list)
            for value in _STRONG_DIFFERENCE_ORDERS[field]
        }
        for field in CORE_ACOUSTIC_FIELDS
    }

    choices: list[list[str]] = []
    choice_costs: list[dict[str, float]] = []
    for field_index, field in enumerate(CORE_ACOUSTIC_FIELDS):
        order = _STRONG_DIFFERENCE_ORDERS[field]
        seed_value = seed_values[field_index]
        seed_index = order.index(seed_value)
        field_costs: dict[str, float] = {}
        for value in order:
            level_distance = abs(order.index(value) - seed_index)
            edit_cost = 0.0 if value == seed_value else 1.0 + 0.55 * level_distance
            reuse_cost = 0.18 * usage_counts[field][value]
            field_costs[value] = edit_cost + reuse_cost
        field_choices = [seed_value] if field in locked else list(order)
        field_choices.sort(key=lambda value: field_costs[value])
        choices.append(field_choices)
        choice_costs.append(field_costs)

    def build_candidate(values: tuple[str, ...]) -> BaseVoiceProfile:
        updates = dict(zip(CORE_ACOUSTIC_FIELDS, values))
        roughness = updates["roughness"]
        if roughness in {"medium", "strong"}:
            updates["texture"] = "grainy"
        elif roughness == "none" and seed.texture in {"grainy", "husky"}:
            updates["texture"] = "clean"
        updates["dynamic_range"] = updates["pitch_range"]
        candidate = repair_acoustic_coherence(
            replace(seed, **updates), locked_fields=locked
        )
        if visual_profile is not None:
            candidate = align_voice_to_visual_structure(
                visual_profile, candidate
            )
        return candidate

    def candidate_is_valid(candidate: BaseVoiceProfile) -> bool:
        return profile_meets_contrast(
            candidate,
            reference_list,
            minimum_differences=minimum_differences,
            minimum_strong_differences=minimum_strong_differences,
            minimum_weighted_distance=minimum_weighted_distance,
            minimum_timbre_differences=minimum_timbre_differences,
            minimum_primary_timbre_differences=(
                minimum_primary_timbre_differences
            ),
            minimum_primary_timbre_strong_differences=(
                minimum_primary_timbre_strong_differences
            ),
            minimum_primary_timbre_groups=minimum_primary_timbre_groups,
            minimum_timbre_skeleton_differences=(
                minimum_timbre_skeleton_differences
            ),
            similar_demographic_minimum_core_differences=(
                similar_demographic_minimum_core_differences
            ),
            minimum_macro_axes=minimum_macro_axes,
            minimum_macro_distance=minimum_macro_distance,
        )

    # A deterministic sampler prevents combinatorial backtracking when a large
    # registry contains many nearly identical seeds. Values with low historical
    # usage are preferred, while the fixed seed keeps reruns reproducible.
    if reference_list:
        random_seed = len(reference_list) * 1009 + sum(
            sum(ord(character) for character in value) for value in seed_values
        )
        generator = random.Random(random_seed)
        ranked_choices: list[list[str]] = []
        for field_index, field in enumerate(CORE_ACOUSTIC_FIELDS):
            if field in locked:
                ranked_choices.append([seed_values[field_index]])
                continue
            ranked_choices.append(
                sorted(
                    choices[field_index],
                    key=lambda value: (
                        usage_counts[field][value],
                        choice_costs[field_index][value],
                        value,
                    ),
                )
            )
        for _ in range(6_000):
            sampled_values = tuple(
                field_choices[generator.randrange(len(field_choices))]
                for field_choices in ranked_choices
            )
            sampled_candidate = build_candidate(sampled_values)
            if candidate_is_valid(sampled_candidate):
                return sampled_candidate
        if len(reference_list) >= 8:
            # Do not fall back to exponential search for a large registry.
            # The caller can report the collision or retry model redesign,
            # while the batch remains bounded and continues predictably.
            return None

    selected_profile: BaseVoiceProfile | None = None
    difference_counts = [0] * len(reference_values)
    strong_counts = [0] * len(reference_values)
    selected: list[str] = []

    def coherence_penalty(values: tuple[str, ...]) -> float:
        signature = dict(zip(CORE_ACOUSTIC_FIELDS, values))
        penalty = 0.0
        if signature["resonance_position"] in {"head", "front"} and signature[
            "resonance_depth"
        ] == "deep":
            penalty += 3.0
        if signature["resonance_position"] == "chest" and signature[
            "resonance_depth"
        ] == "shallow":
            penalty += 3.0
        if signature["vocal_weight"] == "heavy" and signature[
            "resonance_position"
        ] == "head":
            penalty += 3.0
        if signature["vocal_weight"] == "very_light" and signature[
            "resonance_position"
        ] == "chest":
            penalty += 3.0
        if (
            signature["onset"] == "firm"
            and signature["vocal_tension"] == "relaxed"
        ) or (
            signature["onset"] == "soft"
            and signature["vocal_tension"] == "tense"
        ):
            penalty += 2.0
        return penalty

    def search(field_index: int, accumulated_cost: float) -> bool:
        nonlocal selected_profile
        remaining = len(CORE_ACOUSTIC_FIELDS) - field_index
        if any(
            differences + remaining < minimum_differences
            or strong + remaining < minimum_strong_differences
            for differences, strong in zip(difference_counts, strong_counts)
        ):
            return False
        if field_index == len(CORE_ACOUSTIC_FIELDS):
            values = tuple(selected)
            candidate = build_candidate(values)
            if not candidate_is_valid(candidate):
                return False
            # Field choices are already sorted by edit and reuse cost. The
            # first coherent valid design is therefore a close, deterministic
            # fallback and avoids exhaustive optimization as registries grow.
            if coherence_penalty(values) >= 3.0:
                return False
            selected_profile = candidate
            return True

        field = CORE_ACOUSTIC_FIELDS[field_index]
        order = _STRONG_DIFFERENCE_ORDERS[field]
        for value in choices[field_index]:
            changed_references: list[tuple[int, bool]] = []
            for reference_index, reference_signature in enumerate(reference_values):
                reference_value = reference_signature[field_index]
                if value == reference_value:
                    continue
                is_strong = abs(order.index(value) - order.index(reference_value)) >= 2
                difference_counts[reference_index] += 1
                if is_strong:
                    strong_counts[reference_index] += 1
                changed_references.append((reference_index, is_strong))
            selected.append(value)
            found = search(
                field_index + 1,
                accumulated_cost + choice_costs[field_index][value],
            )
            selected.pop()
            for reference_index, is_strong in changed_references:
                difference_counts[reference_index] -= 1
                if is_strong:
                    strong_counts[reference_index] -= 1
            if found:
                return True
        return False

    search(0, 0.0)
    return selected_profile


_BASE_OVERRIDE_CHOICES: dict[str, tuple[str, ...]] = {
    "voice_presentation": VOICE_PRESENTATIONS,
    "pitch_center": PITCH_CENTERS,
    "pitch_range": PITCH_RANGES,
    "vocal_weight": VOCAL_WEIGHTS,
    "brightness": BRIGHTNESSES,
    "resonance_position": RESONANCE_POSITIONS,
    "resonance_depth": RESONANCE_DEPTHS,
    "breathiness": BREATHINESS_LEVELS,
    "nasality": NASALITY_LEVELS,
    "texture": TEXTURES,
    "roughness": ROUGHNESS_LEVELS,
    "onset": ONSETS,
    "articulation": ARTICULATIONS,
    "vocal_tension": VOCAL_TENSIONS,
    "warmth": WARMTH_LEVELS,
    "vocal_distance": VOCAL_DISTANCES,
    "maturity": MATURITY_LEVELS,
    "vocal_maturity": VOCAL_MATURITIES,
    "speech_rate": BASE_SPEECH_RATES,
    "dynamic_range": DYNAMIC_RANGES,
    "pause_pattern": PAUSE_PATTERNS,
    "rhythm_style": RHYTHM_STYLES,
    "ending_style": ENDING_STYLES,
}


def apply_voice_overrides(
    base: BaseVoiceProfile, overrides: Mapping[str, Any]
) -> BaseVoiceProfile:
    """Apply only valid, explicitly returned stable-voice overrides."""
    updates: dict[str, Any] = {}
    if "visual_age_style" in overrides and overrides["visual_age_style"] is not None:
        updates["visual_age_style"] = _visual_age(overrides["visual_age_style"])

    for field_name, allowed in _BASE_OVERRIDE_CHOICES.items():
        value = overrides.get(field_name)
        if value is None:
            continue
        normalized = str(value).strip()
        if normalized in allowed:
            updates[field_name] = normalized

    if "overall_voice_style" in overrides:
        selected_styles = _style_choices(overrides.get("overall_voice_style"))
        if selected_styles:
            updates["overall_voice_style"] = selected_styles

    if "avoidances" in overrides:
        updates["avoidances"] = _list_choices(
            overrides.get("avoidances"), AVOIDANCES, (), 2
        )
    elif updates:
        # Image-derived negative constraints can directly contradict an explicit
        # age, gender or timbre request, so do not retain them after an override.
        updates["avoidances"] = ()

    return replace(base, **updates) if updates else base


def infer_literal_voice_overrides(instruction: str) -> dict[str, Any]:
    """Guarantee common explicit Chinese voice constraints survive model parsing."""
    text = instruction.strip()
    overrides: dict[str, Any] = {}

    age_match = re.search(r"(?<!\d)(\d{1,2})\s*岁", text)
    if age_match:
        requested_age = int(age_match.group(1))
        overrides["visual_age_style"] = requested_age
        overrides["maturity"] = (
            "youthful"
            if requested_age <= 18
            else "mature"
            if requested_age >= MATURE_ADULT_AGE
            else "young"
        )
        overrides["vocal_maturity"] = (
            "youthful"
            if requested_age <= 18
            else "young_adult"
            if requested_age <= 25
            else "mature_young"
            if requested_age < MATURE_ADULT_AGE
            else "mature"
            if requested_age <= 55
            else "seasoned"
        )
    elif "成年" in text:
        overrides["maturity"] = "mature"

    if re.search(r"少女感|少年感|稚嫩声线", text):
        overrides["vocal_maturity"] = "youthful"
    elif "年轻成年" in text:
        overrides["vocal_maturity"] = "young_adult"
    elif re.search(r"成熟青年|较成熟的青年", text):
        overrides["vocal_maturity"] = "mature_young"
    elif re.search(r"老练|岁月感", text):
        overrides["vocal_maturity"] = "seasoned"

    if re.search(r"男性|男声|男人声音|男人的声音", text):
        overrides["voice_presentation"] = "masculine"
    elif re.search(r"女性|女声|女人声音|女人的声音", text):
        overrides["voice_presentation"] = "feminine"
    elif re.search(r"中性声线|中性声音|中性嗓音", text):
        overrides["voice_presentation"] = "androgynous"

    if "鼻音" in text:
        if re.search(r"没有鼻音|无鼻音|不要鼻音", text):
            overrides["nasality"] = "none"
        elif re.search(r"鼻音.{0,3}(很重|浓重|强烈|明显|重)", text):
            overrides["nasality"] = "strong"
        elif re.search(r"(很重|浓重|强烈|明显|重).{0,3}鼻音", text):
            overrides["nasality"] = "strong"
        elif re.search(r"(轻微|略带|轻).{0,3}鼻音|鼻音.{0,3}(轻微|略带|轻)", text):
            overrides["nasality"] = "slight"
        else:
            overrides["nasality"] = "medium"

    requested_styles: list[str] = []
    if re.search(r"粗犷|粗粝|粗砺", text):
        overrides.update(
            pitch_center="medium_low",
            vocal_weight="full",
            brightness="slightly_dark",
            resonance_position="chest",
            resonance_depth="medium",
            breathiness="slight",
            texture="grainy",
            roughness="strong",
            onset="firm",
            articulation="firm",
            vocal_tension="tense",
            vocal_distance="projected",
        )
        requested_styles.extend(["粗犷", "坚定"])
    if re.search(r"浑厚|厚重|厚实|粗厚", text):
        overrides.update(
            pitch_center="medium_low",
            vocal_weight="heavy",
            brightness="slightly_dark",
            resonance_position="chest",
            resonance_depth="deep",
        )
        if "沉稳" not in requested_styles:
            requested_styles.append("沉稳")
    if re.search(r"没有颗粒感|无颗粒感|不要粗糙|不要粗粝", text):
        overrides["roughness"] = "none"
    elif re.search(r"(明显|很强|强烈|浓重).{0,3}(颗粒感|粗糙|粗粝)", text):
        overrides["roughness"] = "strong"
    elif re.search(r"(轻微|略带|轻).{0,3}(颗粒感|粗糙|粗粝)", text):
        overrides["roughness"] = "slight"
    if re.search(r"低沉|低音", text):
        overrides["pitch_center"] = "low"
        overrides["brightness"] = "dark"
    if requested_styles:
        overrides["overall_voice_style"] = requested_styles[:3]

    return overrides


EMOTIONS = ("自然", "愤怒", "悲伤", "温柔", "喜悦", "恐惧", "紧张", "冷静", "严肃", "兴奋", "疲惫", "讽刺")
INTENSITIES = ("轻微", "适中", "强烈")
PACES = ("很慢", "略慢", "适中", "略快", "很快")
FORCES = ("很轻", "偏轻", "适中", "较强", "强烈")
INTONATIONS = ("平稳", "略有起伏", "起伏明显", "起伏强烈")
DELIVERY_ARTICULATIONS = ("轻柔", "自然清晰", "清晰有力", "有力", "略带停顿", "略带颤抖")
STATES = ("自然", "克制", "压抑", "坚定", "爆发", "哽咽", "松弛", "急切")


@dataclass(frozen=True)
class DeliveryProfile:
    emotion: str = "自然"
    intensity: str = "适中"
    pace: str = "适中"
    force: str = "适中"
    intonation: str = "平稳"
    articulation: str = "自然清晰"
    state: str = "自然"

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "DeliveryProfile":
        return cls(
            emotion=_choice(data.get("emotion"), EMOTIONS, "自然"),
            intensity=_choice(data.get("intensity"), INTENSITIES, "适中"),
            pace=_choice(data.get("pace"), PACES, "适中"),
            force=_choice(data.get("force"), FORCES, "适中"),
            intonation=_choice(data.get("intonation"), INTONATIONS, "平稳"),
            articulation=_choice(data.get("articulation"), DELIVERY_ARTICULATIONS, "自然清晰"),
            state=_choice(data.get("state"), STATES, "自然"),
        )


_PRESENTATION_LABELS = {"feminine": "女性", "masculine": "男性", "androgynous": "中性"}
_MATURITY_LABELS = {"youthful": "偏稚嫩", "young": "年轻", "mature": "成熟"}
_VOCAL_MATURITY_LABELS = {
    "young_adult": "年轻成年感",
    "mature_young": "较成熟的青年感",
    "mature": "成熟感",
    "seasoned": "老练与岁月感",
}
_PITCH_CENTER_LABELS = {
    "very_low": "很低", "low": "偏低", "medium_low": "中低",
    "medium": "中等", "medium_high": "中高", "high": "偏高",
}
_PITCH_RANGE_LABELS = {"narrow": "偏窄", "moderate": "适中", "wide": "较宽"}
_VOCAL_WEIGHT_LABELS = {
    "very_light": "极轻薄", "light": "轻薄", "medium": "厚度中等",
    "full": "饱满", "heavy": "厚重",
}
_BRIGHTNESS_LABELS = {
    "dark": "偏暗", "slightly_dark": "略暗", "neutral": "适中",
    "bright": "较高", "very_bright": "很高",
}
_RESONANCE_POSITION_LABELS = {
    "head": "以头腔共鸣为主",
    "front": "以前置共鸣为主，带轻微头腔色彩",
    "mixed": "以混合共鸣为主",
    "chest": "以胸腔共鸣为主",
}
_RESONANCE_DEPTH_LABELS = {
    "shallow": "共鸣较浅", "medium": "共鸣深度适中", "deep": "共鸣较深",
}
_BREATHINESS_LABELS = {
    "none": "几乎没有气声", "slight": "带轻微气声",
    "medium": "带中等程度的柔和气声", "strong": "气声较强",
}
_NASALITY_LABELS = {
    "none": "几乎没有鼻音", "slight": "略带鼻音",
    "medium": "鼻音较明显", "strong": "带浓重鼻音",
}
_TEXTURE_LABELS = {
    "clean": "干净", "silky": "丝滑细腻", "airy": "通透轻盈",
    "husky": "略带沙哑", "grainy": "带颗粒感",
    "metallic": "略带金属感", "crisp": "清脆利落",
}
_ROUGHNESS_LABELS = {
    "none": "表面平滑、没有粗糙感", "slight": "带轻微粗糙颗粒",
    "medium": "粗糙颗粒较明显", "strong": "具有强烈粗粝颗粒感",
}
_ONSET_LABELS = {"soft": "柔和", "balanced": "平衡自然", "firm": "坚定利落"}
_ARTICULATION_LABELS = {
    "soft": "自然偏轻", "natural": "自然", "crisp": "清晰利落", "firm": "清晰有力",
}
_TENSION_LABELS = {
    "relaxed": "较低且放松", "neutral": "适中", "tense": "偏高且紧实",
}
_WARMTH_LABELS = {"cool": "偏冷", "neutral": "中性", "warm": "温暖"}
_DISTANCE_LABELS = {
    "intimate": "贴近而亲密", "natural": "自然", "projected": "外放、有投射感",
}
_BASE_SPEECH_RATE_LABELS = {
    "slow": "很慢", "medium_slow": "略慢", "medium": "适中",
    "medium_fast": "略快", "fast": "很快",
}
_DYNAMIC_RANGE_LABELS = {
    "narrow": "收束", "moderate": "适中", "wide": "宽广",
}
_PAUSE_PATTERN_LABELS = {
    "sparse": "停顿较少",
    "regular": "停顿规律",
    "deliberate": "字句间留有思考感停顿",
    "long": "字句间停顿较长",
    "abrupt": "常有短促的突停",
}
_RHYTHM_STYLE_LABELS = {
    "even": "句子节奏平稳",
    "clipped": "句子短促分明",
    "flowing": "句子连贯流动",
    "elastic": "节奏张弛有层次",
    "syncopated": "节奏跳跃错落",
}
_ENDING_STYLE_LABELS = {
    "rising": "句尾自然上扬",
    "falling": "句尾明确下沉",
    "contained": "句尾干净收束",
    "prolonged": "句尾略有拖长",
}
_INTENSITY_LABELS = {"轻微": "轻微", "适中": "", "强烈": "强烈"}
_STATE_LABELS = {
    "自然": "", "克制": "表达较为克制", "压抑": "表达带有压抑感",
    "坚定": "表达坚定", "爆发": "表达具有爆发感", "哽咽": "表达略带哽咽感",
    "松弛": "表达松弛", "急切": "表达带有急切感",
}


def _join_naturally(items: tuple[str, ...]) -> str:
    if len(items) == 1:
        return items[0]
    return "、".join(items[:-1]) + "而" + items[-1]


def _vocal_maturity_label(base: BaseVoiceProfile) -> str:
    if base.vocal_maturity != "youthful":
        return _VOCAL_MATURITY_LABELS[base.vocal_maturity]
    if base.voice_presentation == "feminine":
        return "少女感"
    if base.voice_presentation == "masculine":
        return "少年感"
    return "青春稚嫩感"


def _default_delivery(base: BaseVoiceProfile) -> DeliveryProfile:
    """Derive a restrained, character-consistent delivery when none is requested."""
    styles = set(base.overall_voice_style)
    if "温柔" in styles or "柔和" in styles:
        emotion = "温柔"
    elif styles & {"清冷", "冷静", "克制", "内收"}:
        emotion = "冷静"
    elif styles & {"明快", "活泼"}:
        emotion = "喜悦"
    else:
        emotion = "自然"

    pace = _BASE_SPEECH_RATE_LABELS[base.speech_rate]

    if (
        base.vocal_distance == "projected"
        or base.vocal_weight in {"full", "heavy"}
        or base.onset == "firm"
    ):
        force = "较强"
    elif (
        base.vocal_distance == "intimate"
        or base.vocal_weight in {"very_light", "light"}
        or base.onset == "soft"
    ):
        force = "偏轻"
    else:
        force = "适中"

    intonation = {
        "narrow": "平稳",
        "moderate": "略有起伏",
        "wide": "起伏明显",
    }[base.pitch_range]
    articulation = {
        "soft": "轻柔",
        "natural": "自然清晰",
        "crisp": "清晰有力",
        "firm": "有力",
    }[base.articulation]
    if styles & {"克制", "内收"}:
        state = "克制"
    elif styles & {"坚定", "有控制力"} or base.vocal_tension == "tense":
        state = "坚定"
    elif styles & {"慵懒", "随性"} or base.vocal_tension == "relaxed":
        state = "松弛"
    else:
        state = "自然"
    return DeliveryProfile(
        emotion=emotion,
        pace=pace,
        force=force,
        intonation=intonation,
        articulation=articulation,
        state=state,
    )


def _build_persona_context(persona: PersonaProfile) -> str:
    """Keep only a compact role anchor in the final TTS prompt.

    Persona lore is useful upstream for deriving speaking behavior, but sending
    the full story to VoiceDesign dilutes the acoustic controls.
    """
    personality = "、".join(persona.core_personality[:2])
    return f"角色定位为{persona.identity}，性格{personality}。"


def build_instruct(
    base: BaseVoiceProfile,
    delivery: DeliveryProfile | None = None,
    persona: PersonaProfile | None = None,
) -> str:
    """Render a compact VoiceDesign prompt dominated by audible attributes."""
    age_and_voice = (
        f"{base.visual_age_style}岁左右的"
        f"{_MATURITY_LABELS[base.maturity]}{_PRESENTATION_LABELS[base.voice_presentation]}声线"
        f"，带有{_vocal_maturity_label(base)}"
    )

    acoustic_parts = [
        age_and_voice,
        (
            f"音调{_PITCH_CENTER_LABELS[base.pitch_center]}，"
            f"音高变化{_PITCH_RANGE_LABELS[base.pitch_range]}"
        ),
        (
            f"声体{_VOCAL_WEIGHT_LABELS[base.vocal_weight]}，"
            f"明亮度{_BRIGHTNESS_LABELS[base.brightness]}"
        ),
        (
            f"音色{_TEXTURE_LABELS[base.texture]}，"
            f"{_ROUGHNESS_LABELS[base.roughness]}"
        ),
        (
            f"{_RESONANCE_POSITION_LABELS[base.resonance_position]}，"
            f"{_RESONANCE_DEPTH_LABELS[base.resonance_depth]}"
        ),
        (
            f"起声{_ONSET_LABELS[base.onset]}，"
            f"张力{_TENSION_LABELS[base.vocal_tension]}，"
            f"咬字{_ARTICULATION_LABELS[base.articulation]}"
        ),
        (
            f"声音{_WARMTH_LABELS[base.warmth]}，"
            f"{_DISTANCE_LABELS[base.vocal_distance]}"
        ),
        (
            f"语速{_BASE_SPEECH_RATE_LABELS[base.speech_rate]}，"
            f"{_PAUSE_PATTERN_LABELS[base.pause_pattern]}，"
            f"{_RHYTHM_STYLE_LABELS[base.rhythm_style]}，"
            f"{_ENDING_STYLE_LABELS[base.ending_style]}"
        ),
    ]

    # Auxiliary coloration is only stated when present. Repeating "no breath /
    # no nasality" across every character creates a strong shared text anchor.
    auxiliaries: list[str] = []
    if base.breathiness != "none":
        auxiliaries.append(_BREATHINESS_LABELS[base.breathiness])
    if base.nasality != "none":
        auxiliaries.append(_NASALITY_LABELS[base.nasality])
    if auxiliaries:
        acoustic_parts.append("，".join(auxiliaries))

    if base.avoidances:
        acoustic_parts.append("同时" + "、".join(base.avoidances))

    effective_delivery = delivery or _default_delivery(base)
    intensity = _INTENSITY_LABELS[effective_delivery.intensity]
    emotion = (
        f"{intensity}{effective_delivery.emotion}"
        if intensity
        else effective_delivery.emotion
    )
    delivery_parts = [
        f"以{emotion}情绪表达",
        f"力度{effective_delivery.force}",
        f"语调{effective_delivery.intonation}",
    ]
    if delivery is not None:
        delivery_parts.append(f"当前咬字{effective_delivery.articulation}")
    state_label = _STATE_LABELS[effective_delivery.state]
    if state_label:
        delivery_parts.append(state_label)

    style_text = _join_naturally(base.overall_voice_style)
    voice_instruct = (
        "；".join(acoustic_parts)
        + "；"
        + "，".join(delivery_parts)
        + f"，整体呈现{style_text}的听觉效果。"
    )
    if persona is None:
        return voice_instruct
    return _build_persona_context(persona) + " 声音设计：" + voice_instruct
