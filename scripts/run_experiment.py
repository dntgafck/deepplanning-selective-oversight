from __future__ import annotations

import fire

try:
    from deepplanning_common import compose_config
    from run_deepplanning_benchmark import run as run_benchmark
except ModuleNotFoundError:
    from scripts.deepplanning_common import compose_config
    from scripts.run_deepplanning_benchmark import run as run_benchmark


def run(
    domain: str | None = None,
    models: str | None = None,
    system: str | None = None,
    parallel: int | None = None,
    max_llm_calls: int | None = None,
    shopping_levels: str | None = None,
    travel_language: str | None = None,
    sample_ids: str | None = None,
) -> None:
    cfg = compose_config(
        "experiment",
        {
            "domain": domain,
            "models": models,
            "system": system,
            "parallel": parallel,
            "max_llm_calls": max_llm_calls,
            "shopping_levels": shopping_levels,
            "travel_language": travel_language,
            "sample_ids": sample_ids,
        },
    )

    domain_name = str(cfg.domain)
    run_benchmark(
        domains=domain_name,
        models=cfg.models,
        system=str(cfg.system),
        shopping_levels=cfg.shopping_levels if domain_name == "shopping" else None,
        shopping_sample_ids=cfg.sample_ids if domain_name == "shopping" else None,
        workers=int(cfg.parallel),
        max_llm_calls=int(cfg.max_llm_calls),
        travel_language=cfg.travel_language if domain_name == "travel" else None,
        travel_sample_ids=cfg.sample_ids if domain_name == "travel" else None,
    )


if __name__ == "__main__":
    fire.Fire(run)
