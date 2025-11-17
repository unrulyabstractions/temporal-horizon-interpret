"""LLM-based prompt pair generation for temporal horizon datasets.

This module generates paired prompts with different temporal horizons using
LLM APIs (OpenAI or Anthropic) to create diverse, high-quality examples.
"""

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from tqdm import tqdm

from .templates import TEMPLATE_LIBRARY, PromptTemplate, format_prompt

logger = logging.getLogger(__name__)


class DatasetGenerator:
    """Generator for temporal horizon prompt pairs.

    Uses LLM APIs to generate diverse prompt pairs with short and long
    temporal horizons based on predefined templates.

    Attributes:
        api_provider: API provider ("openai" or "anthropic")
        api_key: API key for the provider
        model: Model name to use for generation
        temperature: Sampling temperature for generation
    """

    def __init__(
        self,
        api_provider: str = "openai",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
    ):
        """Initialize the dataset generator.

        Args:
            api_provider: API provider ("openai" or "anthropic")
            api_key: API key. If None, will try to get from environment
            model: Model name. If None, uses default for provider
            temperature: Sampling temperature (0.0-1.0)

        Raises:
            ValueError: If api_provider is not supported
            ImportError: If required API client is not installed
        """
        self.api_provider = api_provider.lower()
        self.temperature = temperature

        if self.api_provider == "openai":
            try:
                import openai

                self.client = openai.OpenAI(api_key=api_key)
                self.model = model or "gpt-3.5-turbo"
            except ImportError:
                raise ImportError(
                    "OpenAI package not installed. Install with: pip install openai"
                )
        elif self.api_provider == "anthropic":
            try:
                import anthropic

                self.client = anthropic.Anthropic(api_key=api_key)
                self.model = model or "claude-3-haiku-20240307"
            except ImportError:
                raise ImportError(
                    "Anthropic package not installed. Install with: pip install anthropic"
                )
        else:
            raise ValueError(
                f"Unsupported API provider: {api_provider}. "
                "Supported: ['openai', 'anthropic']"
            )

        logger.info(f"Initialized generator with {api_provider} ({self.model})")

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM API with error handling.

        Args:
            prompt: Prompt to send to the LLM

        Returns:
            Generated text response.

        Raises:
            Exception: If API call fails
        """
        try:
            if self.api_provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=500,
                )
                return response.choices[0].message.content.strip()
            else:  # anthropic
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=500,
                    temperature=self.temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.content[0].text.strip()
        except Exception as e:
            logger.error(f"API call failed: {e}")
            raise

    def generate_task_variation(self, base_task: str, domain: str) -> str:
        """Generate a variation of a task using LLM.

        Args:
            base_task: Base task to vary
            domain: Domain for context

        Returns:
            Varied task description.

        Example:
            >>> gen = DatasetGenerator(api_provider="openai")
            >>> task = gen.generate_task_variation(
            ...     "product launch", "business"
            ... )
            >>> isinstance(task, str)
            True
        """
        prompt = f"""Generate a variation of this {domain} task: "{base_task}"

Requirements:
- Keep it in the same domain ({domain})
- Make it specific and concrete
- Keep it to one short phrase (3-8 words)
- Don't include temporal information
- Return ONLY the task variation, no explanation

Task variation:"""

        return self._call_llm(prompt)

    def generate_prompt_pair(
        self, template: PromptTemplate, task: Optional[str] = None
    ) -> Dict[str, Union[str, int]]:
        """Generate a pair of prompts with different temporal horizons.

        Args:
            template: PromptTemplate to use
            task: Specific task. If None, randomly selects from template examples

        Returns:
            Dictionary containing:
                - short_prompt: Prompt with short temporal horizon
                - long_prompt: Prompt with long temporal horizon
                - domain: Domain category
                - task: Task used
                - label_short: 0 (label for short horizon)
                - label_long: 1 (label for long horizon)

        Example:
            >>> from src.dataset.templates import BUSINESS_TEMPLATES
            >>> gen = DatasetGenerator(api_provider="openai")
            >>> pair = gen.generate_prompt_pair(BUSINESS_TEMPLATES[0])
            >>> "short_prompt" in pair and "long_prompt" in pair
            True
        """
        if task is None:
            task = random.choice(template.task_examples)

        short_prompt = format_prompt(template, task, use_long_horizon=False)
        long_prompt = format_prompt(template, task, use_long_horizon=True)

        return {
            "short_prompt": short_prompt,
            "long_prompt": long_prompt,
            "domain": template.domain,
            "task": task,
            "label_short": 0,
            "label_long": 1,
            "template": template.template,
        }

    def generate(
        self,
        num_pairs: int = 300,
        domains: Optional[List[str]] = None,
        save_path: Optional[Union[str, Path]] = None,
        use_variations: bool = True,
        seed: Optional[int] = 42,
    ) -> List[Dict]:
        """Generate a dataset of prompt pairs.

        Args:
            num_pairs: Number of prompt pairs to generate
            domains: List of domains to include. If None, uses all domains
            save_path: Path to save dataset (JSONL format). If None, doesn't save
            use_variations: If True, generates task variations using LLM
            seed: Random seed for reproducibility

        Returns:
            List of prompt pair dictionaries.

        Example:
            >>> gen = DatasetGenerator(api_provider="openai")
            >>> dataset = gen.generate(num_pairs=10, domains=["business"])
            >>> len(dataset)
            10
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        if domains is None:
            domains = list(TEMPLATE_LIBRARY.keys())

        # Collect all templates
        all_templates = []
        for domain in domains:
            if domain not in TEMPLATE_LIBRARY:
                logger.warning(f"Unknown domain: {domain}, skipping")
                continue
            all_templates.extend(TEMPLATE_LIBRARY[domain])

        if not all_templates:
            raise ValueError(f"No valid templates found for domains: {domains}")

        logger.info(f"Generating {num_pairs} prompt pairs across {len(domains)} domains")

        dataset = []
        for i in tqdm(range(num_pairs), desc="Generating prompt pairs"):
            # Randomly select template
            template = random.choice(all_templates)

            # Select or generate task
            if use_variations and random.random() > 0.5:
                try:
                    base_task = random.choice(template.task_examples)
                    task = self.generate_task_variation(base_task, template.domain)
                except Exception as e:
                    logger.warning(f"Task variation failed, using base: {e}")
                    task = random.choice(template.task_examples)
            else:
                task = random.choice(template.task_examples)

            # Generate pair
            pair = self.generate_prompt_pair(template, task)
            pair["pair_id"] = i
            dataset.append(pair)

        # Save if path provided
        if save_path:
            self._save_dataset(dataset, save_path)

        logger.info(f"Generated {len(dataset)} prompt pairs")
        return dataset

    def _save_dataset(self, dataset: List[Dict], save_path: Union[str, Path]) -> None:
        """Save dataset to JSONL file.

        Args:
            dataset: List of prompt pair dictionaries
            save_path: Path to save file

        Example:
            >>> gen = DatasetGenerator(api_provider="openai")
            >>> dataset = [{"short_prompt": "test", "long_prompt": "test2"}]
            >>> gen._save_dataset(dataset, "/tmp/test.jsonl")
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "w") as f:
            for item in dataset:
                f.write(json.dumps(item) + "\n")

        logger.info(f"Dataset saved to {save_path}")

    def generate_balanced_dataset(
        self,
        num_pairs: int = 300,
        save_path: Optional[Union[str, Path]] = None,
        seed: Optional[int] = 42,
    ) -> List[Dict]:
        """Generate a balanced dataset across all domains.

        Ensures equal representation of each domain in the dataset.

        Args:
            num_pairs: Total number of prompt pairs
            save_path: Path to save dataset
            seed: Random seed

        Returns:
            List of prompt pair dictionaries.

        Example:
            >>> gen = DatasetGenerator(api_provider="openai")
            >>> dataset = gen.generate_balanced_dataset(num_pairs=30)
            >>> len(dataset)
            30
        """
        domains = list(TEMPLATE_LIBRARY.keys())
        pairs_per_domain = num_pairs // len(domains)

        all_pairs = []
        for domain in domains:
            pairs = self.generate(
                num_pairs=pairs_per_domain,
                domains=[domain],
                save_path=None,
                seed=seed,
            )
            all_pairs.extend(pairs)

        # Add remaining pairs to reach exact count
        remaining = num_pairs - len(all_pairs)
        if remaining > 0:
            extra_pairs = self.generate(
                num_pairs=remaining, domains=None, save_path=None, seed=seed
            )
            all_pairs.extend(extra_pairs)

        # Shuffle
        if seed is not None:
            random.seed(seed)
        random.shuffle(all_pairs)

        # Re-index
        for i, pair in enumerate(all_pairs):
            pair["pair_id"] = i

        if save_path:
            self._save_dataset(all_pairs, save_path)

        return all_pairs
