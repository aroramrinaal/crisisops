# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Typed action and observation models for the Crisisops environment."""

from typing import Annotated, Literal, Union

from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, ConfigDict, Field, RootModel


Priority = Literal["low", "normal", "high", "critical"]
RiskLevel = Literal["unknown", "low", "moderate", "high", "critical"]
VerificationStatus = Literal["unverified", "verified", "disputed", "false_alarm"]


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
    risk_level: RiskLevel = "unknown"
    population_estimate: int | None = Field(default=None, ge=0)
    infrastructure_status: dict[
        str, Literal["unknown", "online", "degraded", "offline"]
    ] = Field(default_factory=dict)
    shelter: ShelterInfo | None = None
    routes: list[RouteInfo] = Field(default_factory=list)


class Unit(CrisisopsModel):
    """Limited response resource available to the commander."""

    unit_id: str = Field(..., min_length=1)
    unit_type: Literal[
        "medical", "fire", "rescue", "police", "supply", "recon", "transport"
    ]
    status: Literal["available", "assigned", "en_route", "blocked", "offline"]
    current_zone_id: str | None = None
    capacity: int = Field(default=1, ge=0)
    capabilities: list[str] = Field(default_factory=list)


class Report(CrisisopsModel):
    """Noisy citizen, sensor, official, or field report."""

    report_id: str = Field(..., min_length=1)
    zone_id: str = Field(..., min_length=1)
    source: Literal["citizen", "sensor", "official", "field_team", "media"]
    report_type: Literal[
        "flood",
        "fire",
        "collapse",
        "medical",
        "infrastructure",
        "shelter",
        "resource",
        "other",
    ]
    severity: RiskLevel = "unknown"
    description: str = Field(..., min_length=1)
    verified_status: VerificationStatus = "unverified"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    time_step: int = Field(..., ge=0)


class SitrepPayload(CrisisopsModel):
    """Structured situation report payload for public or operator updates."""

    summary: str = Field(..., min_length=1)
    priorities: list[str] = Field(default_factory=list)
    verified_report_ids: list[str] = Field(default_factory=list)
    pending_verification_report_ids: list[str] = Field(default_factory=list)
    allocations: list[str] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)


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
