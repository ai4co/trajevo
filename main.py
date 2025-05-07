import logging
import os

from pathlib import Path

import hydra

from trajevo import TrajEvo
from utils.utils import init_client, print_hyperlink

ROOT_DIR = os.getcwd()
logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None, config_path="cfg", config_name="config")
def main(cfg):
    workspace_dir = Path.cwd()
    # Set logging level
    logging.info(f"Workspace: {print_hyperlink( workspace_dir)}")
    logging.info(f"Project Root: {print_hyperlink(ROOT_DIR)}")
    logging.info(f"Using LLM: {cfg.get('model', cfg.llm_client.model)}")
    logging.info(f"Using Algorithm: {cfg.algorithm}")

    client = init_client(cfg)
    # optional clients for operators (TrajEvo)
    long_ref_llm = (
        hydra.utils.instantiate(cfg.llm_long_ref) if cfg.get("llm_long_ref") else None
    )
    short_ref_llm = (
        hydra.utils.instantiate(cfg.llm_short_ref) if cfg.get("llm_short_ref") else None
    )
    crossover_llm = (
        hydra.utils.instantiate(cfg.llm_crossover) if cfg.get("llm_crossover") else None
    )
    mutation_llm = (
        hydra.utils.instantiate(cfg.llm_mutation) if cfg.get("llm_mutation") else None
    )

    # Main algorithm
    lhh = TrajEvo(
        cfg,
        ROOT_DIR,
        client,
        long_reflector_llm=long_ref_llm,
        short_reflector_llm=short_ref_llm,
        crossover_llm=crossover_llm,
        mutation_llm=mutation_llm,
        use_wandb=cfg.get("use_wandb", True),  # enable wandb
    )

    best_code_overall, best_code_path_overall = lhh.evolve()
    logging.info(f"Best Code Overall: {best_code_overall}")
    logging.info(f"Best Code Path Overall: {best_code_path_overall}")

    # show all generations best code records
    if hasattr(lhh, "generation_best_records") and lhh.generation_best_records:
        logging.info("==== Each generation best code records ====")
        logging.info("Generation\tScore\tCode Path")
        for gen, record in sorted(lhh.generation_best_records.items()):
            logging.info(f"{gen}\t{record['obj']:.6f}\t{record['code_path']}")
        logging.info("=========================")

    # Run validation and redirect stdout to a file "best_code_overall_stdout.txt"
    with open(f"{ROOT_DIR}/{cfg.problem.problem_name}/gpt.py", "w") as file:
        file.writelines(best_code_overall + "\n")

    # Just print ADE and FDE of the best code overall
    logging.info("Elitist individual (test mode):")
    logging.info(f"ADE: {lhh.elitist['ade_test']}")
    logging.info(f"FDE: {lhh.elitist['fde_test']}")

    if os.path.exists("all_generations_best_records.txt"):
        logging.info(
            f"All generations best records file: {print_hyperlink('all_generations_best_records.txt')}"
        )


if __name__ == "__main__":
    main()
