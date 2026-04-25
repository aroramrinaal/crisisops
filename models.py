# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Typed action and observation models for the Crisisops environment."""

from typing import Annotated, Literal, Union

from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, ConfigDict, Field, RootModel, model_validator


Priority = Literal["low", "normal", "high", "critical"]
RiskLevel = Literal["unknown", "low", "moderate", "high", "critical"]
VerificationStatus = Literal["unverified", "verified", "disputed", "false_alarm"]
IncidentType = Literal[
    "flood",
    "collapse",
    "medical_surge",
    "fire",
    "contamination",
    "power_outage",
]
AccessStatus = Literal["clear", "degraded", "blocked"]
PlanUnitType = Literal[
    "rescue_team",
    "medical_unit",
    "supply_truck",
    "evac_bus",
    "recon_drone",
]
ReportConfidence = Literal[
    "citizen",
    "sensor_confirmed",
    "official_unverified",
]


INCIDENT_REQUIRED_UNIT_TYPES: dict[str, set[str]] = {
    "flood": {"rescue_team", "evac_bus"},
    "collapse": {"rescue_team", "medical_unit"},
    "medical_surge": {"medical_unit"},
    "fire": {"rescue_team"},
    "contamination": {"supply_truck", "medical_unit"},
    "power_outage": {"supply_truck"},
}


class CrisisopsModel(BaseModel):
    """Base model for nested Crisisops payloads."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ShelterInfo(CrisisopsModel):
    """Status and capacity for an emergency shelter."""

    shelter_id: str = Field(..., min_length=1)
    zone_id: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    status: Literal["closed", "opening", "open", "full", "offline"] = "closed"
    capacity_total: int = Field(..., ge=0)
    capacity_available: int = Field(..., ge=0)
    supplies: dict[str, int] = Field(default_factory=dict)


class RouteInfo(CrisisopsModel):
    """Route state between two operational zones."""

    route_id: str = Field(..., min_length=1)
    from_zone_id: str = Field(..., min_length=1)
    to_zone_id: str = Field(..., min_length=1)
    status: Literal["unknown", "open", "congested", "blocked", "unsafe"] = "unknown"
    travel_time_minutes: int | None = Field(default=None, ge=0)
    hazards: list[str] = Field(default_factory=list)


class Zone(CrisisopsModel):
    """Visible map zone and its current incident-command state."""

    zone_id: str = Field(..., min_length=1)
    name: str | None = None
    incident_type: IncidentType
    severity: int = Field(..., ge=1, le=5)
    population_at_risk: int = Field(..., ge=0)
    deadline_steps: int = Field(..., ge=1)
    access_status: AccessStatus = "clear"
    required_unit_types: set[str] = Field(default_factory=set)
    district_id: str | None = None
    risk_level: RiskLevel = "unknown"
    population_estimate: int | None = Field(default=None, ge=0)
    infrastructure_status: dict[
        str, Literal["unknown", "online", "degraded", "offline"]
    ] = Field(default_factory=dict)
    shelter: ShelterInfo | None = None
    routes: list[RouteInfo] = Field(default_factory=list)

    @model_validator(mode="after")
    def derive_plan_fields(self) -> "Zone":
        if not self.required_unit_types:
            self.required_unit_types = set(
                INCIDENT_REQUIRED_UNIT_TYPES[self.incident_type]
            )
        if self.risk_level == "unknown":
            self.risk_level = _risk_for_severity(self.severity)
        if self.population_estimate is None:
            self.population_estimate = self.population_at_risk
        return self


class Unit(CrisisopsModel):
    """Limited response resource available to the commander."""

    unit_id: str = Field(..., min_length=1)
    unit_type: PlanUnitType
    status: Literal["available", "assigned", "en_route", "blocked", "offline"]
    current_zone_id: str | None = None
    travel_cost: int = Field(default=1, ge=0)
    fatigue: int = Field(default=0, ge=0)
    capacity: int = Field(default=1, ge=0)
    capabilities: list[str] = Field(default_factory=list)
    district_id: str | None = None
    shared_pool: bool = False
    mutual_aid_unlock_step: int | None = Field(default=None, ge=0)


class Report(CrisisopsModel):
    """Noisy citizen, sensor, official, or field report."""

    report_id: str = Field(..., min_length=1)
    zone_id: str = Field(..., min_length=1)
    source: Literal["citizen", "sensor", "official", "field_team", "media"]
    report_type: IncidentType
    severity: RiskLevel = "unknown"
    description: str = Field(..., min_length=1)
    verified_status: VerificationStatus = "unverified"
    confidence: ReportConfidence = "citizen"
    time_step: int = Field(..., ge=0)
    reveal_at_step: int = Field(default=0, ge=0)


class SitrepPayload(CrisisopsModel):
    """Structured situation report payload for public or operator updates."""

    incidents_confirmed: list[str] = Field(default_factory=list)
    incidents_resolved: list[str] = Field(default_factory=list)
    unresolved_risks: list[str] = Field(default_factory=list)
    false_alarms_detected: list[str] = Field(default_factory=list)
    summary_text: str = Field(..., min_length=1)


class VerifyReportAction(Action):
    """Verify a noisy report before acting on it."""

    type: Literal["verify_report"]
    report_id: str = Field(..., min_length=1)
    verification_method: Literal[
        "cross_check",
        "contact_source",
        "field_recon",
        "sensor_review",
        "official_confirmation",
    ]
    rationale: str = Field(..., min_length=1)


class RequestReconAction(Action):
    """Request reconnaissance for a zone or report."""

    type: Literal["request_recon"]
    zone_id: str = Field(..., min_length=1)
    objective: str = Field(..., min_length=1)
    priority: Priority = "normal"
    report_id: str | None = None


class AllocateUnitAction(Action):
    """Allocate a response unit to a zone and task."""

    type: Literal["allocate_unit"]
    unit_id: str = Field(..., min_length=1)
    zone_id: str = Field(..., min_length=1)
    task: Literal[
        "rescue",
        "medical",
        "evacuation",
        "fire_suppression",
        "supply_delivery",
        "recon",
        "route_clearance",
    ]
    priority: Priority = "normal"
    report_ids: list[str] = Field(default_factory=list)


class RerouteUnitAction(Action):
    """Move a unit through an alternate route."""

    type: Literal["reroute_unit"]
    unit_id: str = Field(..., min_length=1)
    route: RouteInfo
    reason: str = Field(..., min_length=1)


class IssueEvacuationAction(Action):
    """Issue an evacuation directive for a zone."""

    type: Literal["issue_evacuation"]
    zone_id: str = Field(..., min_length=1)
    urgency: Priority = "high"
    message: str = Field(..., min_length=1)
    route_id: str | None = None
    destination_shelter_id: str | None = None


class OpenShelterAction(Action):
    """Open or update a shelter for evacuees."""

    type: Literal["open_shelter"]
    shelter: ShelterInfo
    reason: str = Field(..., min_length=1)


class DispatchSuppliesAction(Action):
    """Dispatch supplies to a zone or shelter."""

    type: Literal["dispatch_supplies"]
    supplies: dict[str, int] = Field(..., min_length=1)
    destination_zone_id: str = Field(..., min_length=1)
    priority: Priority = "normal"
    unit_id: str | None = None
    destination_shelter_id: str | None = None


class FlagFalseAlarmAction(Action):
    """Mark a report as a false alarm with evidence."""

    type: Literal["flag_false_alarm"]
    report_id: str = Field(..., min_length=1)
    rationale: str = Field(..., min_length=1)
    evidence: list[str] = Field(default_factory=list)


class PublishSitrepAction(Action):
    """Publish a structured situation report."""

    type: Literal["publish_sitrep"]
    payload: SitrepPayload


class NoopAction(Action):
    """Intentionally pause instead of taking an unsafe action."""

    type: Literal["noop"]
    reason: str = Field(..., min_length=1)


CrisisopsActionPayload = Annotated[
    Union[
        VerifyReportAction,
        RequestReconAction,
        AllocateUnitAction,
        RerouteUnitAction,
        IssueEvacuationAction,
        OpenShelterAction,
        DispatchSuppliesAction,
        FlagFalseAlarmAction,
        PublishSitrepAction,
        NoopAction,
    ],
    Field(discriminator="type"),
]


class CrisisopsAction(RootModel[CrisisopsActionPayload]):
    """Discriminated union over every command action accepted by Crisisops."""


class CrisisopsObservation(Observation):
    """Structured commander observation exposed at every timestep."""

    visible_zones: list[Zone] = Field(default_factory=list)
    reports: list[Report] = Field(default_factory=list)
    resources: list[Unit] = Field(default_factory=list)
    time_step: int = Field(default=0, ge=0)
    incident_log: list[str] = Field(default_factory=list)
    session_id: str = Field(..., min_length=1)


def _risk_for_severity(severity: int) -> RiskLevel:
    if severity >= 5:
        return "critical"
    if severity == 4:
        return "high"
    if severity == 3:
        return "moderate"
    return "low"
