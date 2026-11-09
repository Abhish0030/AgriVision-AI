"""Strict response contract and deterministic grounding against local facts."""
import json
import re
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

Text = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=700)]
ShortList = Annotated[list[Text], Field(max_length=8)]


class StrictObject(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Visual(StrictObject):
    visible_symptoms: ShortList
    image_quality: Text
    possible_disease_indicators: ShortList


class CNN(StrictObject):
    efficientnet_prediction: Text
    mobilenet_prediction: Text
    models_agree: bool


class Assessment(StrictObject):
    possible_match: Text
    agreement_with_cnn: Literal["agree", "disagree", "uncertain"]
    explanation: Text


class Reliability(StrictObject):
    status: Literal["uncertain", "review_recommended"]
    warning: Text


class AgentReport(StrictObject):
    visual_analysis: Visual
    cnn_analysis: CNN
    agent_assessment: Assessment
    reliability: Reliability
    recommendations: ShortList
    information_sources: ShortList


def validate_response(raw, context, private_values=()):
    if len(raw) > 24000:
        raise ValueError("oversized_report")
    # Fail closed if output echoes private material or includes markup/URLs in prose.
    lowered = raw.casefold()
    for value in private_values:
        if value and (value.casefold() in lowered or any(
                value[i:i + 48].casefold() in lowered for i in range(0, max(0, len(value) - 47)))):
            raise ValueError("private_material")
    def unique_object(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError("duplicate_field")
            obj[key] = value
        return obj
    report = AgentReport.model_validate(json.loads(raw, object_pairs_hook=unique_object))
    data = report.model_dump()
    expected = context["cnn_analysis"]
    if data["cnn_analysis"] != expected:
        raise ValueError("altered_cnn_fields")
    assessment = data["agent_assessment"]
    allowed = {p["class"] for result in context["predictions"].values() for p in result["top_three"]}
    if assessment["possible_match"] not in allowed | {"uncertain"}:
        raise ValueError("unsupported_match")
    primary = context["primary_class"]
    match = assessment["possible_match"]
    agreement = "uncertain" if match == "uncertain" else "agree" if match == primary else "disagree"
    if assessment["agreement_with_cnn"] != agreement:
        raise ValueError("inconsistent_agreement")
    facts = context["verified_information"]
    prevention = {item for fact in facts for item in fact.get("prevention", [])}
    sources = {fact["source_url"] for fact in facts}
    if not set(data["recommendations"]).issubset(prevention) or not set(data["information_sources"]).issubset(sources):
        raise ValueError("ungrounded_advice_or_source")
    prose = json.dumps([data["visual_analysis"], assessment, data["reliability"]])
    if re.search(r"https?://|<|>|\b(?:authorization|bearer|api[_ -]?key|system[_ -]?prompt)\b", prose, re.I):
        raise ValueError("unsafe_prose")
    return data
