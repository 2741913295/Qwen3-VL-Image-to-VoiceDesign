import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from run_voice_design import (
    load_reference_designs,
    load_reference_profiles,
    save_instruct,
    update_profile_registry,
)
from voice_design.inference import VoiceDesigner, extract_json_object, text_content
from voice_design.profiles import (
    BaseVoiceProfile,
    DeliveryProfile,
    PersonaProfile,
    SpeakingBehavior,
    VisualAnchorReview,
    VisualProfile,
    VoiceDesignReference,
    acoustic_contrast,
    align_voice_to_visual_structure,
    allocate_contrasting_profile,
    apply_voice_overrides,
    build_instruct,
    infer_literal_voice_overrides,
    layered_design_is_too_similar,
    macro_voice_contrast,
    primary_timbre_contrast,
    profile_meets_contrast,
    repair_acoustic_coherence,
    reconcile_visual_anchors,
    timbre_identity_contrast,
    weighted_acoustic_distance,
)


SAMPLE_BASE = BaseVoiceProfile.from_mapping(
    {
        "visual_age_style": 23,
        "voice_presentation": "feminine",
        "pitch_center": "medium_high",
        "pitch_range": "narrow",
        "vocal_weight": "light",
        "brightness": "bright",
        "resonance_position": "front",
        "resonance_depth": "shallow",
        "breathiness": "medium",
        "nasality": "none",
        "texture": "silky",
        "onset": "soft",
        "articulation": "soft",
        "vocal_tension": "relaxed",
        "warmth": "cool",
        "vocal_distance": "intimate",
        "maturity": "young",
        "vocal_maturity": "young_adult",
        "character_identity": "理论物理研究生",
        "speech_rate": "medium_slow",
        "dynamic_range": "narrow",
        "pause_pattern": "long",
        "rhythm_style": "clipped",
        "ending_style": "contained",
        "overall_voice_style": ["清冷", "轻盈", "内收"],
        "avoidances": [],
    }
)

SECOND_BASE = BaseVoiceProfile.from_mapping(
    {
        "visual_age_style": 25,
        "voice_presentation": "feminine",
        "pitch_center": "medium",
        "pitch_range": "moderate",
        "vocal_weight": "full",
        "brightness": "bright",
        "resonance_position": "mixed",
        "resonance_depth": "medium",
        "breathiness": "slight",
        "nasality": "slight",
        "texture": "clean",
        "onset": "firm",
        "articulation": "crisp",
        "vocal_tension": "tense",
        "warmth": "cool",
        "vocal_distance": "projected",
        "maturity": "young",
        "vocal_maturity": "mature_young",
        "character_identity": "急诊主治医生",
        "speech_rate": "medium_fast",
        "dynamic_range": "wide",
        "pause_pattern": "abrupt",
        "rhythm_style": "syncopated",
        "ending_style": "falling",
        "overall_voice_style": ["冷静", "干练", "有控制力"],
        "avoidances": ["不要播音主持腔"],
    }
)

SAMPLE_VISUAL = VisualProfile.from_mapping(
    {
        "visual_age_style": 23,
        "gender_presentation": "feminine",
        "facial_maturity": "young",
        "visual_mass": "light",
        "hair_style": "长发线条柔和",
        "facial_hair": "none",
        "clothing_style": "轻盈精致",
        "clothing_weight": "light",
        "posture": "composed",
        "expression": "gentle",
        "character_type": "清冷学者",
        "visual_temperament": ["清冷", "精致", "内收"],
    }
)

SAMPLE_ANCHOR_REVIEW = VisualAnchorReview.from_mapping(
    {
        "age_style": 23,
        "gender_presentation": "feminine",
        "facial_maturity": "young",
        "visual_mass": "light",
        "facial_hair": "none",
        "confidence": "high",
    }
)

SAMPLE_PERSONA = PersonaProfile.from_mapping(
    {
        "gender": "feminine",
        "age_style": 23,
        "identity": "理论物理研究生",
        "core_personality": ["内敛", "偏执", "专注"],
        "archetype": "理性狂热型",
        "moral_tendency": "本能善但手段激进",
        "motivation": "证明无人相信的理论",
        "social_style": "少言且只回应关键问题",
        "emotional_pattern": "平时压抑，受质疑时发紧",
    }
)

SAMPLE_BEHAVIOR = SpeakingBehavior.from_mapping(
    {
        "speech_rate": "medium_slow",
        "pause_pattern": "long",
        "rhythm_style": "clipped",
        "articulation_style": "precise",
        "emotional_reactivity": "low",
        "restraint_style": "inward",
        "sentence_energy": "gentle",
        "ending_style": "contained",
        "vocal_distance": "intimate",
    }
)

SECOND_PERSONA = PersonaProfile.from_mapping(
    {
        "gender": "feminine",
        "age_style": 23,
        "identity": "急诊主治医生",
        "core_personality": ["果断", "务实", "护短"],
        "archetype": "行动守护型",
        "moral_tendency": "优先救人",
        "motivation": "在混乱中维持生命秩序",
        "social_style": "直接下达清晰短指令",
        "emotional_pattern": "危急时更冷静，事后才释放压力",
    }
)

SECOND_BEHAVIOR = SpeakingBehavior.from_mapping(
    {
        "speech_rate": "fast",
        "pause_pattern": "abrupt",
        "rhythm_style": "syncopated",
        "articulation_style": "firm",
        "emotional_reactivity": "very_low",
        "restraint_style": "restrained",
        "sentence_energy": "forceful",
        "ending_style": "falling",
        "vocal_distance": "projected",
    }
)


class ProfileTests(unittest.TestCase):
    def test_visual_profile_preserves_voice_design_evidence(self):
        self.assertEqual(SAMPLE_VISUAL.visual_age_style, 23)
        self.assertEqual(SAMPLE_VISUAL.gender_presentation, "feminine")
        self.assertEqual(SAMPLE_VISUAL.visual_mass, "light")
        self.assertEqual(SAMPLE_VISUAL.character_type, "清冷学者")
        self.assertEqual(len(SAMPLE_VISUAL.visual_temperament), 3)

    def test_bearded_mature_imposing_anchor_cannot_remain_twenty_five(self):
        initial = VisualProfile.from_mapping(
            {
                **SAMPLE_VISUAL.as_json_dict(),
                "visual_age_style": 25,
                "gender_presentation": "masculine",
            }
        )
        review = VisualAnchorReview.from_mapping(
            {
                "age_style": 25,
                "gender_presentation": "masculine",
                "facial_maturity": "mature",
                "visual_mass": "imposing",
                "facial_hair": "full",
                "confidence": "high",
            }
        )

        reconciled = reconcile_visual_anchors(initial, review)

        self.assertEqual(reconciled.visual_age_style, 38)
        self.assertEqual(reconciled.gender_presentation, "masculine")
        self.assertEqual(reconciled.facial_maturity, "mature")

    def test_trimmed_beard_and_mature_angular_face_enters_mature_band(self):
        initial = VisualProfile.from_mapping(
            {
                **SAMPLE_VISUAL.as_json_dict(),
                "visual_age_style": 32,
                "gender_presentation": "masculine",
                "facial_maturity": "mature",
                "facial_contour": "angular",
                "facial_hair": "light",
                "expression": "stern",
            }
        )
        review = VisualAnchorReview.from_mapping(
            {
                "age_style": 32,
                "gender_presentation": "masculine",
                "facial_maturity": "mature",
                "visual_mass": "balanced",
                "facial_contour": "angular",
                "body_silhouette": "slender",
                "styling_structure": "structured",
                "facial_hair": "light",
                "confidence": "high",
            }
        )

        reconciled = reconcile_visual_anchors(initial, review)

        self.assertEqual(reconciled.visual_age_style, 35)
        self.assertEqual(reconciled.facial_maturity, "mature")

    def test_mature_bearded_masculine_visual_repairs_young_voice_controls(self):
        visual = VisualProfile.from_mapping(
            {
                **SAMPLE_VISUAL.as_json_dict(),
                "visual_age_style": 35,
                "gender_presentation": "masculine",
                "facial_maturity": "mature",
                "facial_hair": "light",
                "facial_contour": "angular",
            }
        )
        young_voice = BaseVoiceProfile.from_mapping(
            {
                **SAMPLE_BASE.as_json_dict(),
                "visual_age_style": 35,
                "voice_presentation": "masculine",
                "maturity": "young",
                "vocal_maturity": "young_adult",
                "pitch_center": "medium_high",
                "vocal_weight": "light",
                "brightness": "bright",
                "resonance_position": "head",
                "resonance_depth": "shallow",
                "avoidances": ["不要成熟大叔感"],
            }
        )

        repaired = align_voice_to_visual_structure(visual, young_voice)

        self.assertEqual(repaired.maturity, "mature")
        self.assertEqual(repaired.vocal_maturity, "mature")
        self.assertEqual(repaired.pitch_center, "medium")
        self.assertEqual(repaired.vocal_weight, "medium")
        self.assertEqual(repaired.resonance_position, "mixed")
        self.assertEqual(repaired.resonance_depth, "medium")
        self.assertNotIn("不要成熟大叔感", repaired.avoidances)

    def test_persona_profile_is_separate_from_acoustic_fingerprint(self):
        persona_json = SAMPLE_PERSONA.as_json_dict()
        voice_json = SAMPLE_BASE.as_json_dict()

        self.assertEqual(persona_json["identity"], "理论物理研究生")
        self.assertEqual(persona_json["archetype"], "理性狂热型")
        self.assertNotIn("pitch_center", persona_json)
        self.assertNotIn("speech_rate", persona_json)
        self.assertNotIn("visual_mass", voice_json)
        self.assertNotIn("clothing_style", voice_json)

    def test_persona_prose_is_compact_and_not_cut_mid_clause(self):
        profile = PersonaProfile.from_mapping(
            {
                **SAMPLE_PERSONA.as_json_dict(),
                "identity": "边关铁甲将军，后面跟着冗长世界观设定",
                "social_style": "不与同僚闲谈，只在战前下达明确命令，战后保持沉默并拒绝解释所有决定",
                "core_personality": ["果决", "言如铁锤", "纪律严明"],
            }
        )

        self.assertEqual(profile.identity, "边关铁甲将军")
        self.assertEqual(profile.social_style, "不与同僚闲谈，只在战前下达明确命令")
        self.assertEqual(profile.core_personality, ("果决", "纪律严明"))

    def test_poetic_worldbuilding_persona_is_flagged_and_not_used_as_traits(self):
        profile = PersonaProfile.from_mapping(
            {
                **SAMPLE_PERSONA.as_json_dict(),
                "identity": "断剑遗孤·星陨剑使",
                "core_personality": ["以残锋为镜", "照见众生虚妄"],
                "moral_tendency": "剑非为杀，为斩断命运之锁",
            }
        )

        self.assertNotIn("以残锋为镜", profile.core_personality)
        self.assertGreater(profile.quality_issue_count(), 0)

    def test_speaking_behavior_is_derived_before_acoustics(self):
        behavior_json = SAMPLE_BEHAVIOR.as_json_dict()

        self.assertEqual(behavior_json["pause_pattern"], "long")
        self.assertEqual(behavior_json["ending_style"], "contained")
        self.assertNotIn("pitch_center", behavior_json)
        self.assertNotIn("vocal_weight", behavior_json)

    def test_final_instruction_can_include_fictional_persona_context(self):
        result = build_instruct(SAMPLE_BASE, persona=SAMPLE_PERSONA)

        self.assertIn("角色定位为理论物理研究生", result)
        self.assertIn("性格内敛、偏执", result)
        self.assertIn("声音设计：23岁左右的年轻女性声线", result)
        self.assertNotIn("证明无人相信的理论", result)
        self.assertNotIn("少言且只回应关键问题", result)
        self.assertLess(len(result), 360)

    def test_unseen_character_identity_is_not_restricted_to_an_enum(self):
        identity = "蒸汽时代落魄发明家"
        profile = BaseVoiceProfile.from_mapping({"character_identity": identity})

        self.assertEqual(profile.character_identity, identity)
        self.assertEqual(
            BaseVoiceProfile.from_mapping(
                {"character_identity": "戴眼镜穿西装的男性"}
            ).character_identity,
            "未明确角色",
        )

    def test_open_ended_voice_styles_keep_audio_terms_and_drop_visual_terms(self):
        profile = BaseVoiceProfile.from_mapping(
            {"overall_voice_style": ["狡黠", "市井感", "戴眼镜的斯文感"]}
        )

        self.assertEqual(profile.overall_voice_style, ("狡黠", "市井感"))

    def test_all_fingerprint_dimensions_are_rendered(self):
        result = build_instruct(SAMPLE_BASE)
        required_phrases = (
            "23岁左右的年轻女性声线",
            "音调中高",
            "音高变化偏窄",
            "声体轻薄",
            "明亮度较高",
            "音色丝滑细腻",
            "以前置共鸣为主",
            "共鸣较浅",
            "起声柔和",
            "张力较低且放松",
            "咬字自然偏轻",
            "语速略慢",
            "字句间停顿较长",
        )
        for phrase in required_phrases:
            self.assertIn(phrase, result)

    def test_expression_is_appended_without_replacing_fingerprint(self):
        delivery = DeliveryProfile.from_mapping(
            {
                "emotion": "愤怒",
                "intensity": "强烈",
                "pace": "略快",
                "force": "较强",
                "intonation": "起伏明显",
                "articulation": "有力",
                "state": "爆发",
            }
        )
        result = build_instruct(SAMPLE_BASE, delivery)
        base_identity = build_instruct(SAMPLE_BASE).split("；以", maxsplit=1)[0]
        self.assertTrue(result.startswith(base_identity + "；"))
        self.assertIn("以强烈愤怒情绪表达", result)
        self.assertIn("当前咬字有力", result)
        self.assertIn("整体呈现清冷、轻盈而内收的听觉效果", result)

    def test_final_instruction_contains_no_visual_or_schema_terms(self):
        result = build_instruct(SAMPLE_BASE)
        forbidden_terms = (
            "眼镜",
            "长发",
            "服饰",
            "妆容",
            "姿态",
            "pitch_center",
            "overall_voice_style",
            "JSON",
            "理论物理研究生",
        )
        for term in forbidden_terms:
            self.assertNotIn(term, result)

    def test_literal_voice_constraint_overrides_image_identity_and_timbre(self):
        instruction = "声音需要听起来是一个成年45岁男性的粗犷并且鼻音很重的声音。"
        overrides = infer_literal_voice_overrides(instruction)
        result_profile = apply_voice_overrides(SAMPLE_BASE, overrides)
        result = build_instruct(result_profile)

        self.assertEqual(result_profile.visual_age_style, 45)
        self.assertEqual(result_profile.voice_presentation, "masculine")
        self.assertEqual(result_profile.maturity, "mature")
        self.assertEqual(result_profile.vocal_maturity, "mature")
        self.assertEqual(result_profile.nasality, "strong")
        self.assertEqual(result_profile.pitch_center, "medium_low")
        self.assertEqual(result_profile.vocal_weight, "full")
        self.assertEqual(result_profile.resonance_position, "chest")
        self.assertEqual(result_profile.texture, "grainy")
        self.assertEqual(result_profile.roughness, "strong")
        self.assertIn("45岁左右的成熟男性声线", result)
        self.assertIn("带浓重鼻音", result)
        self.assertIn("粗犷", result)

    def test_rough_and_thick_constraint_forms_a_coherent_heavy_voice(self):
        overrides = infer_literal_voice_overrides("声音需要听起来像粗犷浑厚的说话方式")
        result_profile = apply_voice_overrides(SAMPLE_BASE, overrides)
        result = build_instruct(result_profile)

        self.assertEqual(result_profile.pitch_center, "medium_low")
        self.assertEqual(result_profile.vocal_weight, "heavy")
        self.assertEqual(result_profile.resonance_position, "chest")
        self.assertEqual(result_profile.resonance_depth, "deep")
        self.assertEqual(result_profile.roughness, "strong")
        self.assertIn("音调中低", result)
        self.assertIn("声体厚重", result)

    def test_expression_only_instruction_has_no_literal_voice_override(self):
        self.assertEqual(infer_literal_voice_overrides("使用非常悲伤的语气说"), {})

    def test_two_visual_designs_differ_across_core_dimensions(self):
        core_fields = (
            "pitch_center",
            "pitch_range",
            "vocal_weight",
            "brightness",
            "resonance_position",
            "resonance_depth",
            "breathiness",
            "nasality",
            "texture",
            "onset",
            "articulation",
            "vocal_tension",
            "warmth",
            "vocal_distance",
        )
        differences = sum(
            getattr(SAMPLE_BASE, field) != getattr(SECOND_BASE, field)
            for field in core_fields
        )
        self.assertGreaterEqual(differences, 6)
        self.assertNotEqual(build_instruct(SAMPLE_BASE), build_instruct(SECOND_BASE))

    def test_age_and_character_name_do_not_count_as_acoustic_contrast(self):
        renamed = BaseVoiceProfile.from_mapping(
            {
                **SAMPLE_BASE.as_json_dict(),
                "visual_age_style": 45,
                "character_identity": "完全不同的角色名称",
            }
        )

        self.assertEqual(acoustic_contrast(renamed, SAMPLE_BASE), (0, 0))
        self.assertFalse(profile_meets_contrast(renamed, [SAMPLE_BASE]))

    def test_young_feminine_voices_need_four_high_weight_differences(self):
        only_three = BaseVoiceProfile.from_mapping(
            {
                **SAMPLE_BASE.as_json_dict(),
                "vocal_maturity": "mature_young",
                "pitch_center": "low",
                "vocal_weight": "heavy",
            }
        )
        four_core = BaseVoiceProfile.from_mapping(
            {
                **only_three.as_json_dict(),
                "resonance_position": "chest",
                "pitch_range": "wide",
                "roughness": "strong",
                "vocal_tension": "tense",
            }
        )

        self.assertFalse(profile_meets_contrast(only_three, [SAMPLE_BASE]))
        self.assertTrue(profile_meets_contrast(four_core, [SAMPLE_BASE]))

    def test_auxiliary_breath_and_nasality_have_lower_similarity_weight(self):
        auxiliary_change = BaseVoiceProfile.from_mapping(
            {
                **SAMPLE_BASE.as_json_dict(),
                "breathiness": "strong",
                "nasality": "strong",
            }
        )
        one_core_change = BaseVoiceProfile.from_mapping(
            {**SAMPLE_BASE.as_json_dict(), "pitch_center": "low"}
        )

        self.assertEqual(weighted_acoustic_distance(auxiliary_change, SAMPLE_BASE), 0.5)
        self.assertEqual(weighted_acoustic_distance(one_core_change, SAMPLE_BASE), 2.0)

    def test_prosody_changes_cannot_hide_same_core_timbre(self):
        prosody_only = BaseVoiceProfile.from_mapping(
            {
                **SAMPLE_BASE.as_json_dict(),
                "pitch_range": "wide",
                "onset": "firm",
                "articulation": "firm",
                "vocal_tension": "tense",
                "speech_rate": "fast",
                "dynamic_range": "wide",
                "pause_pattern": "abrupt",
                "rhythm_style": "syncopated",
                "ending_style": "falling",
            }
        )

        self.assertEqual(timbre_identity_contrast(prosody_only, SAMPLE_BASE), 0)
        self.assertFalse(profile_meets_contrast(prosody_only, [SAMPLE_BASE]))

    def test_raw_field_changes_inside_same_macro_voice_family_do_not_pass(self):
        same_family = BaseVoiceProfile.from_mapping(
            {
                **SAMPLE_BASE.as_json_dict(),
                "vocal_maturity": "seasoned",
                "pitch_center": "high",
                "vocal_weight": "very_light",
                "brightness": "very_bright",
                "resonance_position": "head",
                "articulation": "firm",
                "vocal_tension": "tense",
            }
        )

        macro_axes, _ = macro_voice_contrast(same_family, SAMPLE_BASE)
        self.assertLess(macro_axes, 3)
        self.assertFalse(profile_meets_contrast(same_family, [SAMPLE_BASE]))

    def test_four_phonation_changes_cannot_hide_same_voice_body_and_surface(self):
        too_close = BaseVoiceProfile.from_mapping(
            {
                **SAMPLE_BASE.as_json_dict(),
                "pitch_center": "low",
                "brightness": "dark",
                "onset": "firm",
                "vocal_tension": "tense",
                "articulation": "firm",
                "speech_rate": "fast",
                "pause_pattern": "abrupt",
            }
        )
        separated = BaseVoiceProfile.from_mapping(
            {
                **too_close.as_json_dict(),
                "vocal_weight": "heavy",
                "resonance_position": "chest",
                "resonance_depth": "deep",
                "texture": "grainy",
                "roughness": "strong",
            }
        )

        self.assertGreaterEqual(primary_timbre_contrast(too_close, SAMPLE_BASE)[0], 4)
        self.assertFalse(profile_meets_contrast(too_close, [SAMPLE_BASE]))
        self.assertGreaterEqual(
            primary_timbre_contrast(separated, SAMPLE_BASE)[0], 4
        )
        self.assertTrue(profile_meets_contrast(separated, [SAMPLE_BASE]))

    def test_visual_structure_repairs_inverted_voice_weight_and_projection(self):
        grounded_visual = VisualProfile.from_mapping(
            {
                **SAMPLE_VISUAL.as_json_dict(),
                "visual_mass": "balanced",
                "clothing_weight": "heavy",
                "posture": "upright",
                "expression": "stern",
            }
        )
        grounded = align_voice_to_visual_structure(grounded_visual, SAMPLE_BASE)

        self.assertEqual(grounded.vocal_weight, "medium")
        self.assertEqual(grounded.resonance_depth, "medium")
        self.assertEqual(grounded.vocal_distance, "projected")
        self.assertEqual(grounded.onset, "balanced")

        inverted_light_voice = BaseVoiceProfile.from_mapping(
            {
                **SAMPLE_BASE.as_json_dict(),
                "vocal_weight": "heavy",
                "resonance_position": "chest",
                "resonance_depth": "deep",
                "vocal_distance": "projected",
                "onset": "firm",
                "vocal_tension": "tense",
            }
        )
        light = align_voice_to_visual_structure(SAMPLE_VISUAL, inverted_light_voice)

        self.assertEqual(light.vocal_weight, "medium")
        self.assertEqual(light.resonance_position, "mixed")
        self.assertEqual(light.resonance_depth, "medium")
        self.assertEqual(light.vocal_distance, "natural")
        self.assertEqual(light.onset, "balanced")
        self.assertEqual(light.vocal_tension, "neutral")

    def test_independent_visual_structure_creates_audible_timbre_separation(self):
        broad_structured = VisualProfile.from_mapping(
            {
                **SAMPLE_VISUAL.as_json_dict(),
                "visual_mass": "solid",
                "facial_contour": "broad",
                "body_silhouette": "broad",
                "styling_structure": "structured",
                "clothing_weight": "heavy",
                "posture": "upright",
            }
        )
        slender_armored = VisualProfile.from_mapping(
            {
                **SAMPLE_VISUAL.as_json_dict(),
                "visual_mass": "balanced",
                "facial_contour": "angular",
                "body_silhouette": "slender",
                "styling_structure": "armored",
                "clothing_weight": "heavy",
                "posture": "dynamic",
            }
        )
        full_chest_voice = BaseVoiceProfile.from_mapping(
            {
                **SAMPLE_BASE.as_json_dict(),
                "vocal_weight": "full",
                "brightness": "dark",
                "resonance_position": "chest",
                "resonance_depth": "deep",
                "texture": "clean",
                "roughness": "none",
            }
        )

        broad_voice = align_voice_to_visual_structure(
            broad_structured, full_chest_voice
        )
        slender_voice = align_voice_to_visual_structure(
            slender_armored, full_chest_voice
        )

        self.assertEqual(broad_voice.vocal_weight, "full")
        self.assertEqual(broad_voice.resonance_position, "chest")
        self.assertEqual(slender_voice.vocal_weight, "medium")
        self.assertEqual(slender_voice.resonance_position, "mixed")
        self.assertEqual(slender_voice.resonance_depth, "medium")
        self.assertEqual(slender_voice.texture, "metallic")
        self.assertGreaterEqual(
            primary_timbre_contrast(slender_voice, broad_voice)[0], 4
        )

    def test_acoustic_coherence_repairs_contradictory_enum_combinations(self):
        contradictory = BaseVoiceProfile.from_mapping(
            {
                **SAMPLE_BASE.as_json_dict(),
                "vocal_weight": "heavy",
                "resonance_position": "head",
                "resonance_depth": "shallow",
                "texture": "clean",
                "roughness": "strong",
                "onset": "firm",
                "vocal_tension": "relaxed",
            }
        )

        repaired = repair_acoustic_coherence(contradictory)

        self.assertEqual(repaired.resonance_position, "mixed")
        self.assertEqual(repaired.resonance_depth, "medium")
        self.assertEqual(repaired.texture, "grainy")
        self.assertEqual(repaired.vocal_tension, "neutral")

    def test_layered_similarity_detects_same_character_design(self):
        reference = VoiceDesignReference(
            SAMPLE_BASE, SAMPLE_PERSONA, SAMPLE_BEHAVIOR
        )

        self.assertTrue(
            layered_design_is_too_similar(
                SAMPLE_PERSONA, SAMPLE_BEHAVIOR, SAMPLE_BASE, reference
            )
        )

    def test_distinct_persona_prevents_three_layer_collapse(self):
        distinct_persona = PersonaProfile.from_mapping(
            {
                **SAMPLE_PERSONA.as_json_dict(),
                "identity": "急诊主治医生",
                "core_personality": ["果断", "务实", "护短"],
                "archetype": "行动守护型",
                "moral_tendency": "优先救人",
                "motivation": "在混乱中维持生命秩序",
                "social_style": "直接下达清晰短指令",
                "emotional_pattern": "危急时更冷静，事后才释放压力",
            }
        )
        reference = VoiceDesignReference(
            SAMPLE_BASE, SAMPLE_PERSONA, SAMPLE_BEHAVIOR
        )

        self.assertFalse(
            layered_design_is_too_similar(
                distinct_persona, SAMPLE_BEHAVIOR, SAMPLE_BASE, reference
            )
        )

    def test_allocator_finds_nearest_valid_acoustic_fallback(self):
        allocated = allocate_contrasting_profile(SAMPLE_BASE, [SAMPLE_BASE])

        if allocated is None:
            self.fail("allocator unexpectedly returned None")
        self.assertTrue(profile_meets_contrast(allocated, [SAMPLE_BASE]))
        self.assertEqual(allocated.visual_age_style, SAMPLE_BASE.visual_age_style)
        self.assertEqual(allocated.voice_presentation, SAMPLE_BASE.voice_presentation)
        self.assertEqual(allocated.character_identity, SAMPLE_BASE.character_identity)

    def test_allocator_respects_locked_explicit_voice_fields(self):
        allocated = allocate_contrasting_profile(
            SAMPLE_BASE,
            [SAMPLE_BASE],
            locked_fields=("pitch_center", "vocal_weight", "resonance_position"),
        )

        if allocated is None:
            self.fail("allocator unexpectedly returned None")
        self.assertEqual(allocated.pitch_center, SAMPLE_BASE.pitch_center)
        self.assertEqual(allocated.vocal_weight, SAMPLE_BASE.vocal_weight)
        self.assertEqual(
            allocated.resonance_position, SAMPLE_BASE.resonance_position
        )

    def test_allocator_can_fill_a_six_character_batch_from_same_seed(self):
        allocated_profiles = [SAMPLE_BASE]
        for _ in range(5):
            allocated = allocate_contrasting_profile(
                SAMPLE_BASE, allocated_profiles
            )
            if allocated is None:
                self.fail("allocator unexpectedly returned None")
            self.assertTrue(
                profile_meets_contrast(allocated, allocated_profiles)
            )
            allocated_profiles.append(allocated)

    def test_defaults_are_detected_as_overly_neutral(self):
        base = BaseVoiceProfile.from_mapping({})
        self.assertGreater(base.neutral_dimension_count(), 3)
        self.assertEqual(base.avoidances, ())
        self.assertEqual(len(base.overall_voice_style), 2)

    def test_age_is_rounded_to_nearest_supported_anchor(self):
        base = BaseVoiceProfile.from_mapping({"visual_age_style": "34岁左右"})
        self.assertEqual(base.visual_age_style, 35)

    def test_extract_json_from_fenced_output(self):
        payload = {"emotion": "悲伤", "pace": "略慢"}
        wrapped = "```json\n" + json.dumps(payload, ensure_ascii=False) + "\n```"
        self.assertEqual(extract_json_object(wrapped), payload)

    def test_text_content_uses_qwen_multimodal_message_format(self):
        self.assertEqual(
            text_content("系统提示"),
            [{"type": "text", "text": "系统提示"}],
        )

    def test_instruction_is_saved_with_image_stem(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = save_instruct("测试声线。", "/data/人物样例.png", temp_dir)
            self.assertEqual(output_path.name, "人物样例.txt")
            self.assertEqual(output_path.read_text(encoding="utf-8"), "测试声线。\n")

    def test_profile_registry_excludes_the_current_image_on_rerun(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            registry = root / ".voice_profiles.json"
            first_image = root / "人物一.png"
            second_image = root / "人物二.png"
            update_profile_registry(str(registry), str(first_image), SAMPLE_BASE)

            references = load_reference_profiles(str(registry), str(second_image))
            own_references = load_reference_profiles(str(registry), str(first_image))

        self.assertEqual(references, [SAMPLE_BASE])
        self.assertEqual(own_references, [])

    def test_profile_registry_round_trips_all_three_design_layers(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            registry = root / ".voice_profiles.json"
            first_image = root / "人物一.png"
            second_image = root / "人物二.png"
            update_profile_registry(
                str(registry),
                str(first_image),
                SAMPLE_BASE,
                SAMPLE_PERSONA,
                SAMPLE_BEHAVIOR,
            )

            references = load_reference_designs(
                str(registry), str(second_image)
            )

        self.assertEqual(
            references,
            [VoiceDesignReference(SAMPLE_BASE, SAMPLE_PERSONA, SAMPLE_BEHAVIOR)],
        )


class _FakeInputs(dict):
    def __init__(self):
        super().__init__(input_ids=[[1, 2]])
        self.input_ids = self["input_ids"]

    def to(self, _device):
        return self


class _FakeProcessor:
    load_kwargs: dict[str, Any] = {}

    def __init__(self):
        self.messages = []
        self.outputs = [
            json.dumps(SAMPLE_VISUAL.as_json_dict(), ensure_ascii=False),
            json.dumps(SAMPLE_ANCHOR_REVIEW.as_json_dict(), ensure_ascii=False),
            json.dumps(SAMPLE_PERSONA.as_json_dict(), ensure_ascii=False),
            json.dumps(SAMPLE_BEHAVIOR.as_json_dict(), ensure_ascii=False),
            json.dumps(SAMPLE_BASE.as_json_dict(), ensure_ascii=False),
            json.dumps(
                {
                    "emotion": "悲伤",
                    "intensity": "强烈",
                    "pace": "略慢",
                    "force": "偏轻",
                    "intonation": "起伏明显",
                    "articulation": "略带停顿",
                    "state": "压抑",
                },
                ensure_ascii=False,
            ),
        ]

    @classmethod
    def from_pretrained(cls, _path, **kwargs):
        cls.load_kwargs = kwargs
        return cls()

    def apply_chat_template(self, messages, **_kwargs):
        for message in messages:
            if not isinstance(message["content"], list):
                raise TypeError("message content must be a multimodal content list")
            for content in message["content"]:
                if not isinstance(content, dict) or "type" not in content:
                    raise TypeError("each content item must be a typed object")
        self.messages.append(messages)
        return _FakeInputs()

    def batch_decode(self, *_args, **_kwargs):
        return [self.outputs.pop(0)]


class _FakeModel:
    device = "cpu"
    load_kwargs: dict[str, Any] = {}

    @classmethod
    def from_pretrained(cls, _path, **kwargs):
        cls.load_kwargs = kwargs
        return cls()

    def eval(self):
        return self

    def generate(self, **_kwargs):
        return [[1, 2, 3]]


class _FakeVLLM:
    load_kwargs: dict[str, Any] = {}

    def __init__(self, **kwargs):
        type(self).load_kwargs = kwargs
        self.inputs = []

    def generate(self, inputs, **kwargs):
        self.inputs.append((inputs, kwargs))
        choice = types.SimpleNamespace(text='{"result":"ok"}')
        return [types.SimpleNamespace(outputs=[choice])]


class _FakeVLLMProcessor:
    image_processor = types.SimpleNamespace(patch_size=16)
    load_kwargs: dict[str, Any] = {}

    @classmethod
    def from_pretrained(cls, _path, **kwargs):
        cls.load_kwargs = kwargs
        return cls()

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.template_kwargs = kwargs
        return "official-qwen-rendered-prompt"


class _FakeSamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _RepairingFakeProcessor(_FakeProcessor):
    def __init__(self):
        super().__init__()
        self.outputs = [
            "这不是 JSON",
            json.dumps(SAMPLE_VISUAL.as_json_dict(), ensure_ascii=False),
            json.dumps(SAMPLE_ANCHOR_REVIEW.as_json_dict(), ensure_ascii=False),
            json.dumps(SAMPLE_PERSONA.as_json_dict(), ensure_ascii=False),
            json.dumps(SAMPLE_BEHAVIOR.as_json_dict(), ensure_ascii=False),
            json.dumps(SAMPLE_BASE.as_json_dict(), ensure_ascii=False),
        ]


class _NeutralThenDistinctiveProcessor(_FakeProcessor):
    def __init__(self):
        super().__init__()
        neutral = BaseVoiceProfile.from_mapping({})
        self.outputs = [
            json.dumps(SAMPLE_VISUAL.as_json_dict(), ensure_ascii=False),
            json.dumps(SAMPLE_ANCHOR_REVIEW.as_json_dict(), ensure_ascii=False),
            json.dumps(SAMPLE_PERSONA.as_json_dict(), ensure_ascii=False),
            json.dumps(SAMPLE_BEHAVIOR.as_json_dict(), ensure_ascii=False),
            json.dumps(neutral.as_json_dict(), ensure_ascii=False),
            json.dumps(SAMPLE_BASE.as_json_dict(), ensure_ascii=False),
        ]


class _AlwaysNeutralProcessor(_FakeProcessor):
    def __init__(self):
        super().__init__()
        neutral_json = json.dumps(
            BaseVoiceProfile.from_mapping({}).as_json_dict(), ensure_ascii=False
        )
        self.outputs = [
            json.dumps(SAMPLE_VISUAL.as_json_dict(), ensure_ascii=False),
            json.dumps(SAMPLE_ANCHOR_REVIEW.as_json_dict(), ensure_ascii=False),
            json.dumps(SAMPLE_PERSONA.as_json_dict(), ensure_ascii=False),
            json.dumps(SAMPLE_BEHAVIOR.as_json_dict(), ensure_ascii=False),
            neutral_json,
            neutral_json,
        ]


class _VoiceOverrideFakeProcessor(_FakeProcessor):
    def __init__(self):
        super().__init__()
        self.outputs = [json.dumps({"vocal_weight": "full"}, ensure_ascii=False)]


class _ContrastRepairProcessor(_FakeProcessor):
    def __init__(self):
        super().__init__()
        distinctive = {
            **SAMPLE_BASE.as_json_dict(),
            "pitch_center": "very_low",
            "pitch_range": "wide",
            "vocal_weight": "heavy",
            "brightness": "dark",
            "resonance_position": "chest",
            "resonance_depth": "deep",
            "roughness": "strong",
            "onset": "firm",
            "vocal_tension": "tense",
            "articulation": "firm",
        }
        self.outputs = [
            json.dumps(SECOND_PERSONA.as_json_dict(), ensure_ascii=False),
            json.dumps(SECOND_BEHAVIOR.as_json_dict(), ensure_ascii=False),
            json.dumps(distinctive, ensure_ascii=False),
            json.dumps(distinctive, ensure_ascii=False),
        ]


class _PersistentCollisionProcessor(_FakeProcessor):
    def __init__(self):
        super().__init__()
        repeated_persona = json.dumps(
            SAMPLE_PERSONA.as_json_dict(), ensure_ascii=False
        )
        repeated_behavior = json.dumps(
            SAMPLE_BEHAVIOR.as_json_dict(), ensure_ascii=False
        )
        repeated_voice = json.dumps(SAMPLE_BASE.as_json_dict(), ensure_ascii=False)
        self.outputs = [
            repeated_persona,
            repeated_behavior,
            repeated_voice,
            repeated_persona,
            repeated_behavior,
            repeated_voice,
        ]


class InferenceContractTests(unittest.TestCase):
    def test_vllm_backend_reuses_official_qwen_processor_and_vision_loader(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            vllm_source = Path(temp_dir) / "vllm-0.28.0"
            package_dir = vllm_source / "vllm"
            package_dir.mkdir(parents=True)
            package_init = package_dir / "__init__.py"
            package_init.write_text("# fake vllm", encoding="utf-8")
            fake_vllm = types.SimpleNamespace(
                LLM=_FakeVLLM,
                SamplingParams=_FakeSamplingParams,
                __version__="0.28.0",
                __file__=str(package_init),
            )
            fake_transformers = types.SimpleNamespace(
                AutoProcessor=_FakeVLLMProcessor
            )
            vision_calls = []

            def fake_process_vision_info(messages, **kwargs):
                vision_calls.append((messages, kwargs))
                return (["decoded-local-image"], None, {"fps": 2.0})

            fake_qwen_vl_utils = types.SimpleNamespace(
                process_vision_info=fake_process_vision_info
            )
            image = Path(temp_dir) / "中文人物.png"
            image.write_bytes(b"fake image")
            with mock.patch.dict(
                sys.modules,
                {
                    "vllm": fake_vllm,
                    "transformers": fake_transformers,
                    "qwen_vl_utils": fake_qwen_vl_utils,
                },
            ):
                designer = VoiceDesigner(
                    temp_dir,
                    backend="vllm",
                    allowed_local_media_path=temp_dir,
                    vllm_source_path=vllm_source,
                )
                output = designer._generate(
                    [
                        {"role": "system", "content": text_content("system")},
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": str(image)},
                                {"type": "text", "text": "analyze"},
                            ],
                        },
                    ]
                )

        self.assertEqual(output, '{"result":"ok"}')
        self.assertEqual(_FakeVLLM.load_kwargs["model"], str(Path(temp_dir).resolve()))
        self.assertTrue(_FakeVLLM.load_kwargs["enable_prefix_caching"])
        self.assertEqual(_FakeVLLM.load_kwargs["gpu_memory_utilization"], 0.15)
        self.assertEqual(
            _FakeVLLM.load_kwargs["kv_cache_memory_bytes"], 2 * 1024**3
        )
        request = designer.model.inputs[0][0][0]
        self.assertEqual(request["prompt"], "official-qwen-rendered-prompt")
        self.assertEqual(
            request["multi_modal_data"]["image"], ["decoded-local-image"]
        )
        self.assertEqual(
            request["mm_processor_kwargs"], {"fps": 2.0, "do_resize": False}
        )
        self.assertEqual(vision_calls[0][1]["image_patch_size"], 16)
        self.assertFalse(designer.processor.template_kwargs["tokenize"])
        self.assertEqual(designer.vllm_version, "0.28.0")
        self.assertEqual(designer.vllm_source_path, vllm_source.resolve())

    def test_vllm_backend_rejects_a_different_loaded_version(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            vllm_source = Path(temp_dir) / "vllm-0.28.0"
            package_dir = vllm_source / "vllm"
            package_dir.mkdir(parents=True)
            package_init = package_dir / "__init__.py"
            package_init.write_text("# fake vllm", encoding="utf-8")
            fake_vllm = types.SimpleNamespace(
                LLM=_FakeVLLM,
                SamplingParams=_FakeSamplingParams,
                __version__="0.29.0",
                __file__=str(package_init),
            )
            with mock.patch.dict(sys.modules, {"vllm": fake_vllm}):
                with self.assertRaisesRegex(RuntimeError, "要求 0.28.x"):
                    VoiceDesigner(
                        temp_dir,
                        backend="vllm",
                        allowed_local_media_path=temp_dir,
                        vllm_source_path=vllm_source,
                    )

    def test_vocal_maturity_is_clamped_to_visual_age_band(self):
        youthful_persona = PersonaProfile.from_mapping(
            {**SAMPLE_PERSONA.as_json_dict(), "age_style": 18}
        )
        mature_persona = PersonaProfile.from_mapping(
            {**SAMPLE_PERSONA.as_json_dict(), "age_style": 40}
        )

        self.assertEqual(
            VoiceDesigner._consistent_vocal_maturity("mature", youthful_persona),
            "youthful",
        )
        self.assertEqual(
            VoiceDesigner._consistent_vocal_maturity(
                "young_adult", mature_persona
            ),
            "mature",
        )

    def test_batch_collision_is_redesigned_before_returning(self):
        fake_transformers = types.SimpleNamespace(
            AutoModelForImageTextToText=_FakeModel,
            AutoProcessor=_ContrastRepairProcessor,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            image = Path(temp_dir) / "碰撞人物.png"
            image.write_bytes(b"fake image")
            with mock.patch.dict(sys.modules, {"transformers": fake_transformers}):
                designer = VoiceDesigner(temp_dir)
                _, _, redesigned = designer.ensure_batch_contrast(
                    SAMPLE_BASE,
                    [
                        VoiceDesignReference(
                            SAMPLE_BASE, SAMPLE_PERSONA, SAMPLE_BEHAVIOR
                        )
                    ],
                    visual_profile=SAMPLE_VISUAL,
                    persona_profile=SAMPLE_PERSONA,
                    speaking_behavior=SAMPLE_BEHAVIOR,
                )

        self.assertTrue(profile_meets_contrast(redesigned, [SAMPLE_BASE]))
        self.assertEqual(redesigned.visual_age_style, SAMPLE_BASE.visual_age_style)
        self.assertEqual(redesigned.voice_presentation, SAMPLE_BASE.voice_presentation)

    def test_unresolved_three_layer_collision_uses_final_allocator(self):
        fake_transformers = types.SimpleNamespace(
            AutoModelForImageTextToText=_FakeModel,
            AutoProcessor=_PersistentCollisionProcessor,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            image = Path(temp_dir) / "持续碰撞.png"
            image.write_bytes(b"fake image")
            with mock.patch.dict(sys.modules, {"transformers": fake_transformers}):
                designer = VoiceDesigner(temp_dir)
                _, _, redesigned = designer.ensure_batch_contrast(
                    SAMPLE_BASE,
                    [
                        VoiceDesignReference(
                            SAMPLE_BASE, SAMPLE_PERSONA, SAMPLE_BEHAVIOR
                        )
                    ],
                    visual_profile=SAMPLE_VISUAL,
                    persona_profile=SAMPLE_PERSONA,
                    speaking_behavior=SAMPLE_BEHAVIOR,
                    max_redesign_attempts=2,
                )

        self.assertTrue(profile_meets_contrast(redesigned, [SAMPLE_BASE]))

    def test_sparse_voice_instruction_overrides_only_requested_field(self):
        fake_transformers = types.SimpleNamespace(
            AutoModelForImageTextToText=_FakeModel,
            AutoProcessor=_VoiceOverrideFakeProcessor,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch.dict(sys.modules, {"transformers": fake_transformers}):
                designer = VoiceDesigner(temp_dir)
                updated = designer.apply_voice_instruction(
                    "声音要更加饱满一些", SAMPLE_BASE
                )

        self.assertEqual(updated.vocal_weight, "full")
        self.assertEqual(updated.visual_age_style, SAMPLE_BASE.visual_age_style)
        self.assertEqual(updated.voice_presentation, SAMPLE_BASE.voice_presentation)
        self.assertEqual(updated.pitch_center, SAMPLE_BASE.pitch_center)
        self.assertEqual(len(designer.processor.messages), 1)

    def test_flash_attention_uses_explicit_bfloat16_dtype(self):
        fake_transformers = types.SimpleNamespace(
            AutoModelForImageTextToText=_FakeModel,
            AutoProcessor=_FakeProcessor,
        )
        fake_torch = types.SimpleNamespace(bfloat16="mock-bfloat16")
        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch.dict(
                sys.modules,
                {"transformers": fake_transformers, "torch": fake_torch},
            ):
                VoiceDesigner(temp_dir, attn_implementation="flash_attention_2")

        self.assertEqual(_FakeModel.load_kwargs["dtype"], "mock-bfloat16")
        self.assertEqual(
            _FakeModel.load_kwargs["attn_implementation"], "flash_attention_2"
        )

    def test_local_loading_and_staged_prompt_isolation(self):
        fake_transformers = types.SimpleNamespace(
            AutoModelForImageTextToText=_FakeModel,
            AutoProcessor=_FakeProcessor,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            image = Path(temp_dir) / "徐若溪.png"
            image.write_bytes(b"fake image for mocked processor")
            with mock.patch.dict(sys.modules, {"transformers": fake_transformers}):
                designer = VoiceDesigner(temp_dir)
                base = designer.analyze_base_voice(str(image))
                delivery = designer.analyze_delivery("悲伤、低沉地表达", base)

        self.assertTrue(_FakeProcessor.load_kwargs["local_files_only"])
        self.assertTrue(_FakeModel.load_kwargs["local_files_only"])
        self.assertEqual(base, SAMPLE_BASE)
        if delivery is None:
            self.fail("delivery unexpectedly returned None")
        self.assertEqual(delivery.emotion, "悲伤")
        self.assertEqual(len(designer.processor.messages), 6)
        self.assertNotIn("悲伤、低沉地表达", str(designer.processor.messages[:5]))
        self.assertIn("悲伤、低沉地表达", str(designer.processor.messages[5]))

    def test_plain_unicode_image_path_is_preserved(self):
        fake_transformers = types.SimpleNamespace(
            AutoModelForImageTextToText=_FakeModel,
            AutoProcessor=_FakeProcessor,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            image = Path(temp_dir) / "徐若溪.png"
            image.write_bytes(b"fake image")
            expected = str(image.resolve())
            with mock.patch.dict(sys.modules, {"transformers": fake_transformers}):
                designer = VoiceDesigner(temp_dir)
                designer.analyze_base_voice(str(image))
        actual = designer.processor.messages[0][1]["content"][0]["image"]
        self.assertEqual(actual, expected)
        self.assertFalse(actual.startswith("file://"))
        self.assertNotIn("%", actual)

    def test_overly_neutral_profile_is_refined_with_same_image(self):
        fake_transformers = types.SimpleNamespace(
            AutoModelForImageTextToText=_FakeModel,
            AutoProcessor=_NeutralThenDistinctiveProcessor,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            image = Path(temp_dir) / "徐若薇.png"
            image.write_bytes(b"fake image")
            with mock.patch.dict(sys.modules, {"transformers": fake_transformers}):
                designer = VoiceDesigner(temp_dir)
                profile = designer.analyze_base_voice(str(image))
        self.assertEqual(profile, SAMPLE_BASE)
        self.assertEqual(len(designer.processor.messages), 6)
        self.assertIn("区分度不足", str(designer.processor.messages[5]))

    def test_repeated_overly_neutral_profile_uses_best_available_version(self):
        fake_transformers = types.SimpleNamespace(
            AutoModelForImageTextToText=_FakeModel,
            AutoProcessor=_AlwaysNeutralProcessor,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            image = Path(temp_dir) / "generic.png"
            image.write_bytes(b"fake image")
            with mock.patch.dict(sys.modules, {"transformers": fake_transformers}):
                designer = VoiceDesigner(temp_dir)
                profile = designer.analyze_base_voice(str(image))

        self.assertGreater(profile.neutral_dimension_count(), 3)
        self.assertEqual(len(designer.processor.messages), 6)

    def test_json_repair_messages_also_use_content_lists(self):
        fake_transformers = types.SimpleNamespace(
            AutoModelForImageTextToText=_FakeModel,
            AutoProcessor=_RepairingFakeProcessor,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            image = Path(temp_dir) / "person.jpg"
            image.write_bytes(b"fake image for mocked processor")
            with mock.patch.dict(sys.modules, {"transformers": fake_transformers}):
                designer = VoiceDesigner(temp_dir)
                base = designer.analyze_base_voice(str(image))
        self.assertEqual(base, SAMPLE_BASE)
        self.assertEqual(len(designer.processor.messages), 6)

    def test_blank_instruction_skips_second_generation(self):
        fake_transformers = types.SimpleNamespace(
            AutoModelForImageTextToText=_FakeModel,
            AutoProcessor=_FakeProcessor,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch.dict(sys.modules, {"transformers": fake_transformers}):
                designer = VoiceDesigner(temp_dir)
                delivery = designer.analyze_delivery("  ", SAMPLE_BASE)
        self.assertIsNone(delivery)
        self.assertEqual(designer.processor.messages, [])


if __name__ == "__main__":
    unittest.main()
