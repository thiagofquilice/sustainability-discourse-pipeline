#!/usr/bin/env python3
"""Shared 6-topic discourse catalog for the paper pipeline."""

from __future__ import annotations

from typing import Any


TOPIC_GROUPS: list[dict[str, Any]] = [
    {
        "topic_id": 1,
        "topic_code": "T1",
        "topic_name": "Clean energy transition",
        "topic_definition": (
            "Deployment and expansion of cleaner energy systems, especially renewable electricity, "
            "grid integration, storage, and access to modern clean energy services."
        ),
        "topic_summary_text": (
            "Renewable electricity, low-carbon power generation, clean energy infrastructure, renewable "
            "grid integration, energy storage systems, off-grid renewable electrification, community "
            "access to electricity, access to affordable modern energy services, affordable clean "
            "energy services, community solar systems, shared renewable energy infrastructure, and "
            "rural electrification."
        ),
        "elements": [
            "renewable electricity",
            "low-carbon power generation",
            "clean energy infrastructure",
            "renewable grid integration",
            "energy storage systems",
            "off-grid renewable electrification",
            "community access to electricity",
            "access to affordable modern energy services",
            "affordable clean energy services",
            "community solar systems",
            "shared renewable energy infrastructure",
            "rural electrification",
        ],
        "sdg_crosswalk": ["SDG07", "SDG13"],
        "boundary_notes": (
            "Exclude generic energy finance, generic oil and gas reserves, generic energy markets, and "
            "generic infrastructure spending unless the text clearly concerns cleaner energy transition or energy access."
        ),
        "false_positive_notes": (
            "Common false positives: capital raising for energy firms, oil and gas reserve disclosures, "
            "pipeline capacity, generic utility competition, and generic energy policy with no clean-energy or access content."
        ),
    },
    {
        "topic_id": 2,
        "topic_code": "T2",
        "topic_name": "Operational sustainability and circular production",
        "topic_definition": (
            "Internal operational practices that reduce material, water, energy, chemical, or pollution impacts."
        ),
        "topic_summary_text": (
            "Cleaner production, waste reduction, waste minimization, material recycling, recycling of materials, "
            "material recovery, wastewater reuse, wastewater treatment and reuse, circular material use, hazardous "
            "chemicals management, hazardous waste management, life cycle assessment, industrial waste management, "
            "and secondary recovery projects."
        ),
        "elements": [
            "cleaner production",
            "waste reduction",
            "waste minimization",
            "material recycling",
            "recycling of materials",
            "material recovery",
            "wastewater reuse",
            "wastewater treatment and reuse",
            "circular material use",
            "hazardous chemicals management",
            "hazardous waste management",
            "life cycle assessment",
            "industrial waste management",
            "secondary recovery projects",
        ],
        "sdg_crosswalk": ["SDG12", "SDG06", "SDG09"],
        "boundary_notes": (
            "Focus on internal production, process, and operational impact reduction. Exclude generic climate "
            "regulation and generic sustainability strategy when no concrete operational practice is described."
        ),
        "false_positive_notes": (
            "Common false positives: generic operational efficiency, reserves and processing language, mining "
            "performance, broad ESG reporting, and sustainability claims without concrete operational or process content."
        ),
    },
    {
        "topic_id": 3,
        "topic_code": "T3",
        "topic_name": "Sustainable products, services and consumption",
        "topic_definition": (
            "Lower-impact products, services, materials, and market-facing consumption choices."
        ),
        "topic_summary_text": (
            "Sustainable products, sustainable services, environmentally preferable products, lower-impact products, "
            "lower-impact services, recyclable products, biobased products, lower-impact materials, sustainable "
            "consumption, greener product design, product environmental footprint, and eco-labeled products."
        ),
        "elements": [
            "sustainable products",
            "sustainable services",
            "environmentally preferable products",
            "lower-impact products",
            "lower-impact services",
            "recyclable products",
            "biobased products",
            "lower-impact materials",
            "sustainable consumption",
            "greener product design",
            "product environmental footprint",
            "eco-labeled products",
        ],
        "sdg_crosswalk": ["SDG12", "SDG09"],
        "boundary_notes": (
            "This topic is market-facing. Separate it from T2 by requiring product, service, materials, or consumption "
            "content rather than internal production processes alone."
        ),
        "false_positive_notes": (
            "Common false positives: generic sustainability rhetoric, generic green lifestyle messaging without a concrete "
            "product or consumption practice, and operational efficiency claims with no product or market-facing dimension."
        ),
    },
    {
        "topic_id": 4,
        "topic_code": "T4",
        "topic_name": "Climate strategy, carbon governance and disclosure",
        "topic_definition": (
            "Targets, disclosure, governance, pricing, and formal strategic positioning around climate and carbon."
        ),
        "topic_summary_text": (
            "Greenhouse gas emissions targets, greenhouse gas disclosure, carbon disclosure, internal carbon pricing, "
            "internal carbon price, carbon markets, carbon offsets, climate governance, climate policy position, "
            "executive compensation climate metrics, decarbonization targets, and net-zero targets."
        ),
        "elements": [
            "greenhouse gas emissions targets",
            "greenhouse gas disclosure",
            "carbon disclosure",
            "internal carbon pricing",
            "internal carbon price",
            "carbon markets",
            "carbon offsets",
            "climate governance",
            "climate policy position",
            "executive compensation climate metrics",
            "decarbonization targets",
            "net-zero targets",
        ],
        "sdg_crosswalk": ["SDG13", "SDG12"],
        "boundary_notes": (
            "Require explicit targets, disclosure, governance, or carbon-policy mechanisms. Exclude broad climate concern "
            "or broad ESG language if no formal climate-management mechanism appears."
        ),
        "false_positive_notes": (
            "Common false positives: generic climate concern, generic transition rhetoric, broad sustainability reputation, "
            "or broad investor language without targets, disclosure, pricing, or governance content."
        ),
    },
    {
        "topic_id": 5,
        "topic_code": "T5",
        "topic_name": "Climate risk, adaptation and resilience",
        "topic_definition": (
            "Physical and transition climate risk, climate adaptation, resilience planning, and operational exposure."
        ),
        "topic_summary_text": (
            "Physical climate risk, transition climate risk, climate adaptation, climate resilience planning, disaster "
            "preparedness, extreme weather exposure, climate-related operational impacts, drought adaptation, flood "
            "adaptation, heat adaptation, adaptation to climate change, and climate vulnerability."
        ),
        "elements": [
            "physical climate risk",
            "transition climate risk",
            "climate adaptation",
            "climate resilience planning",
            "disaster preparedness",
            "extreme weather exposure",
            "climate-related operational impacts",
            "drought adaptation",
            "flood adaptation",
            "heat adaptation",
            "adaptation to climate change",
            "climate vulnerability",
        ],
        "sdg_crosswalk": ["SDG13", "SDG11"],
        "boundary_notes": (
            "Focus on exposure, preparedness, resilience, and adaptation. Separate this from T4 by excluding pure "
            "governance/disclosure language when no concrete risk or adaptation content appears."
        ),
        "false_positive_notes": (
            "Common false positives: generic climate change discussion, broad mitigation language, generic energy transition "
            "text, and abstract sustainability risks without concrete climate-risk or adaptation content."
        ),
    },
    {
        "topic_id": 6,
        "topic_code": "T6",
        "topic_name": "Ecosystems, pollution and environmental stewardship",
        "topic_definition": (
            "Protection, restoration, compliance, and pollution-prevention practices tied to biodiversity, land, coast, or ecosystems."
        ),
        "topic_summary_text": (
            "Biodiversity conservation, habitat conservation, habitat restoration, ecological restoration, land degradation "
            "prevention, forest stewardship, biodiversity safeguards, land-use environmental compliance, marine pollution "
            "prevention, spill prevention, discharge reduction, leak control, and coastal ecosystem protection."
        ),
        "elements": [
            "biodiversity conservation",
            "habitat conservation",
            "habitat restoration",
            "ecological restoration",
            "land degradation prevention",
            "forest stewardship",
            "biodiversity safeguards",
            "land-use environmental compliance",
            "marine pollution prevention",
            "spill prevention",
            "discharge reduction",
            "leak control",
            "coastal ecosystem protection",
        ],
        "sdg_crosswalk": ["SDG14", "SDG15", "SDG12"],
        "boundary_notes": (
            "This topic combines ecosystem protection and pollution prevention. Keep only texts with concrete biodiversity, "
            "habitat, stewardship, compliance, or pollution-prevention substance."
        ),
        "false_positive_notes": (
            "Common false positives: generic sustainability claims, broad natural-capital rhetoric, generic environmental "
            "risk language, and marine or land references without stewardship or pollution-prevention substance."
        ),
    },
]


def bullet_list_text(elements: list[str]) -> str:
    return "\n".join(f"- {element}" for element in elements)


def catalog_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in TOPIC_GROUPS:
        rows.append(
            {
                "topic_id": group["topic_id"],
                "topic_code": group["topic_code"],
                "topic_name": group["topic_name"],
                "topic_definition": group["topic_definition"],
                "topic_elements_bullets": bullet_list_text(group["elements"]),
                "topic_embedding_text": group["topic_summary_text"],
                "topic_summary_text": group["topic_summary_text"],
                "sdg_crosswalk": "; ".join(group["sdg_crosswalk"]),
                "boundary_notes": group["boundary_notes"],
                "false_positive_notes": group["false_positive_notes"],
                "element_count": len(group["elements"]),
            }
        )
    return rows
