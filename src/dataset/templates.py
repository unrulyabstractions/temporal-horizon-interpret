"""Prompt templates for temporal horizon dataset generation.

This module provides templates for generating prompt pairs with different temporal
horizons across multiple domains (business, science, personal).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class PromptTemplate:
    """Template for generating temporal horizon prompts.

    Attributes:
        domain: Domain category (business, science, personal)
        template: Base template string with {horizon} and {task} placeholders
        short_horizon: Short temporal scope example (<1 year)
        long_horizon: Long temporal scope example (>1 year)
        task_examples: List of specific tasks for this domain
    """

    domain: str
    template: str
    short_horizon: str
    long_horizon: str
    task_examples: List[str]


# Business domain templates
BUSINESS_TEMPLATES = [
    PromptTemplate(
        domain="business",
        template="Develop a strategic plan for {task} over the {horizon}.",
        short_horizon="next quarter",
        long_horizon="next decade",
        task_examples=[
            "launching a new product line",
            "expanding into new markets",
            "building brand recognition",
            "developing competitive advantages",
            "transforming company culture",
        ],
    ),
    PromptTemplate(
        domain="business",
        template="Create a roadmap for {task} with a {horizon} timeline.",
        short_horizon="6-month",
        long_horizon="10-year",
        task_examples=[
            "digital transformation initiatives",
            "revenue growth targets",
            "market leadership positioning",
            "organizational restructuring",
            "sustainability commitments",
        ],
    ),
    PromptTemplate(
        domain="business",
        template="Outline key milestones for {task} over the {horizon}.",
        short_horizon="next 3 months",
        long_horizon="next 5 years",
        task_examples=[
            "customer acquisition strategy",
            "technology infrastructure upgrades",
            "talent development programs",
            "innovation pipeline development",
            "industry disruption strategy",
        ],
    ),
]

# Science domain templates
SCIENCE_TEMPLATES = [
    PromptTemplate(
        domain="science",
        template="Design a research program to {task} within a {horizon} timeframe.",
        short_horizon="1-year",
        long_horizon="20-year",
        task_examples=[
            "develop new therapeutic approaches",
            "understand climate change impacts",
            "advance quantum computing capabilities",
            "explore space colonization",
            "cure neurodegenerative diseases",
        ],
    ),
    PromptTemplate(
        domain="science",
        template="Propose an experimental approach to {task} over the {horizon}.",
        short_horizon="next 6 months",
        long_horizon="next 15 years",
        task_examples=[
            "validate a novel hypothesis",
            "build a comprehensive theory",
            "develop breakthrough technologies",
            "revolutionize energy production",
            "decode biological systems",
        ],
    ),
    PromptTemplate(
        domain="science",
        template="Create a scientific roadmap for {task} with a {horizon} scope.",
        short_horizon="near-term",
        long_horizon="multi-generational",
        task_examples=[
            "achieving fusion energy",
            "preventing pandemic diseases",
            "reversing environmental damage",
            "enhancing human cognition",
            "establishing interplanetary presence",
        ],
    ),
]

# Personal domain templates
PERSONAL_TEMPLATES = [
    PromptTemplate(
        domain="personal",
        template="Plan your approach to {task} over the {horizon}.",
        short_horizon="next few months",
        long_horizon="rest of your life",
        task_examples=[
            "learning a new skill",
            "building lasting relationships",
            "achieving financial independence",
            "developing expertise in a field",
            "creating a meaningful legacy",
        ],
    ),
    PromptTemplate(
        domain="personal",
        template="Set goals for {task} with a {horizon} perspective.",
        short_horizon="immediate",
        long_horizon="lifetime",
        task_examples=[
            "career advancement",
            "personal growth and development",
            "health and wellness optimization",
            "mastering a craft",
            "making societal impact",
        ],
    ),
    PromptTemplate(
        domain="personal",
        template="Develop a strategy for {task} spanning the {horizon}.",
        short_horizon="next season",
        long_horizon="coming decades",
        task_examples=[
            "building professional network",
            "cultivating wisdom and character",
            "achieving work-life balance",
            "leaving a positive impact",
            "pursuing lifelong learning",
        ],
    ),
]

# Master template dictionary
TEMPLATE_LIBRARY: Dict[str, List[PromptTemplate]] = {
    "business": BUSINESS_TEMPLATES,
    "science": SCIENCE_TEMPLATES,
    "personal": PERSONAL_TEMPLATES,
}


def get_templates(domain: Optional[str] = None) -> List[PromptTemplate]:
    """Get prompt templates for a specific domain or all domains.

    Args:
        domain: Domain to filter by ("business", "science", "personal").
                If None, returns all templates.

    Returns:
        List of PromptTemplate objects.

    Raises:
        ValueError: If domain is not recognized.

    Example:
        >>> templates = get_templates("business")
        >>> len(templates)
        3
        >>> templates = get_templates()
        >>> len(templates)
        9
    """
    if domain is None:
        all_templates = []
        for templates in TEMPLATE_LIBRARY.values():
            all_templates.extend(templates)
        return all_templates

    if domain not in TEMPLATE_LIBRARY:
        raise ValueError(
            f"Unknown domain: {domain}. "
            f"Valid domains: {list(TEMPLATE_LIBRARY.keys())}"
        )

    return TEMPLATE_LIBRARY[domain]


def create_custom_template(
    domain: str,
    template: str,
    short_horizon: str,
    long_horizon: str,
    task_examples: List[str],
) -> PromptTemplate:
    """Create a custom prompt template.

    Args:
        domain: Domain category for the template
        template: Template string with {horizon} and {task} placeholders
        short_horizon: Short temporal scope phrase
        long_horizon: Long temporal scope phrase
        task_examples: List of task examples

    Returns:
        PromptTemplate object.

    Example:
        >>> template = create_custom_template(
        ...     domain="healthcare",
        ...     template="Implement {task} over the {horizon}.",
        ...     short_horizon="next month",
        ...     long_horizon="next generation",
        ...     task_examples=["patient care improvements"]
        ... )
        >>> template.domain
        'healthcare'
    """
    return PromptTemplate(
        domain=domain,
        template=template,
        short_horizon=short_horizon,
        long_horizon=long_horizon,
        task_examples=task_examples,
    )


def format_prompt(template: PromptTemplate, task: str, use_long_horizon: bool) -> str:
    """Format a template into a complete prompt.

    Args:
        template: PromptTemplate to format
        task: Specific task to insert
        use_long_horizon: If True, use long horizon; otherwise use short

    Returns:
        Formatted prompt string.

    Example:
        >>> template = BUSINESS_TEMPLATES[0]
        >>> prompt = format_prompt(template, "product launch", False)
        >>> "next quarter" in prompt
        True
    """
    horizon = template.long_horizon if use_long_horizon else template.short_horizon
    return template.template.format(horizon=horizon, task=task)
