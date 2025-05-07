import logging
import math
import os
import subprocess

from typing import Optional

import numpy as np
import wandb

from omegaconf import DictConfig

from utils.llm_client.base import BaseClient
from utils.utils import (
    block_until_running,
    extract_code_from_generator,
    file_to_string,
    filter_code,
    filter_traceback,
    parse_metrics_from_stdout,
    parse_stats_from_response,
    print_hyperlink,
    set_seed,
)

USE_SLEEP = True  # trick for waiting the logger

logger = logging.getLogger(__name__)


class TrajEvo:
    def __init__(
        self,
        cfg: DictConfig,
        root_dir: str,
        generator_llm: BaseClient,
        reflector_llm: Optional[BaseClient] = None,
        # Support setting different LLMs for each of the four operators:
        # Short-term Reflection, Long-term Reflection, Crossover, Mutation
        short_reflector_llm: Optional[BaseClient] = None,
        long_reflector_llm: Optional[BaseClient] = None,
        crossover_llm: Optional[BaseClient] = None,
        mutation_llm: Optional[BaseClient] = None,
        use_wandb: bool = True,
    ) -> None:
        self.cfg = cfg
        self.generator_llm = generator_llm
        self.reflector_llm = reflector_llm or generator_llm
        self.use_wandb = use_wandb

        self.short_reflector_llm = short_reflector_llm or self.reflector_llm
        self.long_reflector_llm = long_reflector_llm or self.reflector_llm
        self.crossover_llm = crossover_llm or generator_llm
        self.mutation_llm = mutation_llm or generator_llm

        self.root_dir = root_dir

        self.mutation_rate = cfg.mutation_rate
        self.iteration = 0
        self.function_evals = 0
        self.elitist = None
        self.long_term_reflection_str = ""
        self.best_obj_overall = None
        self.best_code_overall = None
        self.best_code_path_overall = None
        # record the best code and score of each generation
        self.generation_best_records = {}

        # record the history best score, for improvement evaluation
        self.best_obj_history = []

        # Set seed
        set_seed(self.cfg.seed)

        if self.use_wandb:
            wandb_project = (
                cfg.wandb_project if hasattr(cfg, "wandb_project") else "trajevo"
            )
            wandb_name = (
                cfg.wandb_name
                if hasattr(cfg, "wandb_name")
                else f"trajevo-{cfg.problem.dataset}"
            )

            def cfg2dict(cfg: DictConfig) -> dict:
                cfg_dict = {}
                for k, v in cfg.items():
                    # if type(v) == DictConfig:
                    if isinstance(v, DictConfig):
                        cfg_dict[k] = cfg2dict(v)
                    else:
                        cfg_dict[k] = v
                return cfg_dict

            wandb.init(project=wandb_project, name=wandb_name, config=cfg2dict(cfg))

        self.init_prompt()
        self.init_population()

    def init_prompt(self) -> None:
        self.problem = self.cfg.problem.problem_name
        self.problem_desc = self.cfg.problem.description
        self.problem_size = self.cfg.problem.problem_size
        self.func_name = self.cfg.problem.func_name
        self.obj_type = self.cfg.problem.obj_type
        self.problem_type = self.cfg.problem.problem_type

        logger.info("Problem: " + self.problem)
        logger.info("Problem description: " + self.problem_desc)
        logger.info("Function name: " + self.func_name)

        self.prompt_dir = f"{self.root_dir}/prompts"
        self.output_file = f"{self.root_dir}/{self.problem}/gpt.py"

        # Loading all text prompts
        # Problem-specific prompt components
        prompt_path_suffix = "_black_box" if self.problem_type == "black_box" else ""
        problem_prompt_path = f"{self.prompt_dir}/{self.problem}{prompt_path_suffix}"
        self.seed_func = file_to_string(f"{problem_prompt_path}/seed_func.txt")
        self.func_signature = file_to_string(f"{problem_prompt_path}/func_signature.txt")
        self.func_desc = file_to_string(f"{problem_prompt_path}/func_desc.txt")
        if os.path.exists(f"{problem_prompt_path}/external_knowledge.txt"):
            self.external_knowledge = file_to_string(
                f"{problem_prompt_path}/external_knowledge.txt"
            )
            self.long_term_reflection_str = self.external_knowledge
        else:
            self.external_knowledge = ""

        # Common prompts
        self.system_generator_prompt = file_to_string(
            f"{self.prompt_dir}/common/system_generator.txt"
        )
        self.system_reflector_prompt = file_to_string(
            f"{self.prompt_dir}/common/system_reflector.txt"
        )
        self.user_reflector_st_prompt = (
            file_to_string(f"{self.prompt_dir}/common/user_reflector_st.txt")
            if self.problem_type != "black_box"
            else file_to_string(
                f"{self.prompt_dir}/common/user_reflector_st_black_box.txt"
            )
        )  # shrot-term reflection
        self.user_reflector_lt_prompt = file_to_string(
            f"{self.prompt_dir}/common/user_reflector_lt.txt"
        )  # long-term reflection
        self.crossover_prompt = file_to_string(f"{self.prompt_dir}/common/crossover.txt")
        self.mutation_prompt = file_to_string(f"{self.prompt_dir}/common/mutation.txt")
        self.user_generator_prompt = file_to_string(
            f"{self.prompt_dir}/common/user_generator.txt"
        ).format(
            func_name=self.func_name,
            problem_desc=self.problem_desc,
            func_desc=self.func_desc,
        )
        self.seed_prompt = file_to_string(f"{self.prompt_dir}/common/seed.txt").format(
            seed_func=self.seed_func,
            func_name=self.func_name,
        )

        # Flag to print prompts
        self.print_crossover_prompt = (
            True  # Print crossover prompt for the first iteration
        )
        self.print_mutate_prompt = True  # Print mutate prompt for the first iteration
        self.print_short_term_reflection_prompt = (
            True  # Print short-term reflection prompt for the first iteration
        )
        self.print_long_term_reflection_prompt = (
            True  # Print long-term reflection prompt for the first iteration
        )

    def init_population(self) -> None:
        # Evaluate the seed function, and set it as Elite
        logger.info("Evaluating seed function...")
        code = extract_code_from_generator(self.seed_func).replace("v1", "v2")
        logger.info("Seed function code: \n" + code)
        seed_ind = {
            "stdout_filepath": f"problem_iter{self.iteration}_stdout0.txt",
            "code_path": f"problem_iter{self.iteration}_code0.py",
            "code": code,
            "response_id": 0,
        }
        self.seed_ind = seed_ind
        self.population = self.evaluate_population([seed_ind])

        # If seed function is invalid, stop
        if not self.seed_ind["exec_success"]:
            raise RuntimeError(
                f"Seed function is invalid. Please check the stdout file in {os.getcwd()}."
            )
        self.seed_ind["stats"] = parse_stats_from_response(
            os.path.abspath(self.seed_ind["stdout_filepath"])
        )

        self.update_iter()

        # Generate responses
        system = self.system_generator_prompt
        user = (
            self.user_generator_prompt
            + "\n"
            + self.seed_prompt
            + "\n"
            + self.long_term_reflection_str
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        logger.info(
            "Initial Population Prompt: \nSystem Prompt: \n"
            + system
            + "\nUser Prompt: \n"
            + user
        )

        responses = self.generator_llm.multi_chat_completion(
            [messages],
            self.cfg.init_pop_size,
            temperature=self.generator_llm.temperature + 0.3,
        )  # Increase the temperature for diverse initial population
        population = [
            self.response_to_individual(response, response_id)
            for response_id, response in enumerate(responses)
        ]

        # Run code and evaluate population
        population = self.evaluate_population(population)

        # Update iteration
        self.population = population
        self.update_iter()

    def response_to_individual(
        self, response: str, response_id: int, file_name: str = None
    ) -> dict:
        """
        Convert response to individual
        """
        # Write response to file
        file_name = (
            f"problem_iter{self.iteration}_response{response_id}.txt"
            if file_name is None
            else file_name + ".txt"
        )
        with open(file_name, "w") as file:
            file.writelines(response + "\n")

        code = extract_code_from_generator(response)

        # Extract code and description from response
        std_out_filepath = (
            f"problem_iter{self.iteration}_stdout{response_id}.txt"
            if file_name is None
            else file_name + "_stdout.txt"
        )

        individual = {
            "stdout_filepath": std_out_filepath,
            "code_path": f"problem_iter{self.iteration}_code{response_id}.py",
            "code": code,
            "response_id": response_id,
            "stats": "unavailable",  # will be updated after evaluation
        }
        return individual

    def mark_invalid_individual(self, individual: dict, traceback_msg: str) -> dict:
        """
        Mark an individual as invalid.
        """
        individual["exec_success"] = False
        individual["obj"] = float("inf")
        individual["traceback_msg"] = traceback_msg
        return individual

    def evaluate_population(self, population: list[dict]) -> list[float]:
        """
        Evaluate population by running code in parallel and computing objective values.
        """
        inner_runs = []
        # Run code to evaluate
        for response_id in range(len(population)):
            self.function_evals += 1
            # Skip if response is invalid
            if population[response_id]["code"] is None:
                population[response_id] = self.mark_invalid_individual(
                    population[response_id], "Invalid response!"
                )
                inner_runs.append(None)
                continue

            logger.info(f"Iteration {self.iteration}: Running Code {response_id}")

            try:
                process = self._run_code(population[response_id], response_id)
                inner_runs.append(process)
            except Exception as e:  # If code execution fails
                logger.error(f"Error for response_id {response_id}: {e}")
                population[response_id] = self.mark_invalid_individual(
                    population[response_id], str(e)
                )
                inner_runs.append(None)

        # Update population with objective values
        for response_id, inner_run in enumerate(inner_runs):
            if inner_run is None:  # If code execution fails, skip
                continue
            try:
                inner_run.communicate(
                    timeout=self.cfg.timeout
                )  # Wait for code execution to finish
            except subprocess.TimeoutExpired as e:
                logger.error(f"Error for response_id {response_id}: {e}")
                population[response_id] = self.mark_invalid_individual(
                    population[response_id], str(e)
                )
                inner_run.kill()
                continue

            individual = population[response_id]
            stdout_filepath = individual["stdout_filepath"]
            with open(stdout_filepath, "r") as f:  # read the stdout file
                stdout_str = f.read()
            traceback_msg = filter_traceback(stdout_str)

            individual = population[response_id]
            # Store objective value for each individual
            if traceback_msg == "":  # If execution has no error
                try:
                    individual["obj"] = (
                        float(stdout_str.split("\n")[-2])
                        if self.obj_type == "min"
                        else -float(stdout_str.split("\n")[-2])
                    )
                    # if obj is nan, set it to inf
                    if np.isnan(individual["obj"]):
                        logger.warning(
                            f"Code worked but returned nan as objective value. Setting to inf. See {print_hyperlink(stdout_filepath, 'stdout')}"
                        )
                        individual["obj"] = float("inf")
                    individual["exec_success"] = True
                    individual["stats"] = parse_stats_from_response(
                        os.path.abspath(individual["stdout_filepath"])
                    )
                except:
                    population[response_id] = self.mark_invalid_individual(
                        population[response_id], "Invalid std out / objective value!"
                    )
            else:  # Otherwise, also provide execution traceback error feedback
                population[response_id] = self.mark_invalid_individual(
                    population[response_id], traceback_msg
                )

            logger.info(
                f"Iteration {self.iteration}, response_id {response_id}: Objective value: {individual['obj']}"
            )
        return population

    def _run_code(
        self,
        individual: dict,
        response_id,
        test: bool = False,
        stdout_filepath: str = None,
    ) -> subprocess.Popen:
        """
        Write code into a file and run eval script.
        """
        logger.debug(f"Iteration {self.iteration}: Processing Code Run {response_id}")

        with open(self.output_file, "w") as file:
            file.writelines(individual["code"] + "\n")

        stdout_filepath = stdout_filepath or individual["stdout_filepath"]

        # Execute the python file with flags
        with open(stdout_filepath, "w") as f:
            eval_file_path = (
                f"{self.root_dir}/{self.problem}/eval.py"
                if self.problem_type != "black_box"
                else f"{self.root_dir}/{self.problem}/eval_black_box.py"
            )
            process = subprocess.Popen(
                [
                    "python",
                    "-u",
                    eval_file_path,
                    "--dataset",
                    self.cfg.problem.dataset,
                    "--test" if test else "--no-test",
                    "--samples_per_scene",
                    str(self.cfg.problem.samples_per_scene),
                    "--samples_per_scene_test",
                    str(self.cfg.problem.samples_per_scene_test),
                    "--num_processes",
                    str(self.cfg.num_processes),
                ],
                stdout=f,
                stderr=f,
            )

        # TODO: this is actually not totally true, check
        block_until_running(
            stdout_filepath,
            log_status=True,
            iter_num=self.iteration,
            response_id=response_id,
        )
        return process

    def update_iter(self) -> None:
        """
        update the status of each generation
        """
        population = self.population
        objs = [individual["obj"] for individual in population]
        best_obj, best_sample_idx = min(objs), np.argmin(np.array(objs))

        # add best objective to history
        self.best_obj_history.append(best_obj)

        # whether to record the best code of current generation
        should_record = True

        # only record the generation with significant improvement
        if self.iteration > 10:
            # get the previous best score
            previous_best = min(
                [record["obj"] for record in self.generation_best_records.values()]
            )

            # calculate the improvement percentage
            improvement = (
                (previous_best - best_obj) / previous_best if previous_best != 0 else 0
            )
            min_improvement_threshold = 0.001  # threshold of improvement

            # if the improvement is not significant, and not every 5 generations
            if improvement < min_improvement_threshold and self.iteration % 5 != 0:
                should_record = False
                logging.info(
                    f"Generation {self.iteration} has no significant improvement ({improvement:.6f}), skipping record"
                )

        # record the best code and score of current generation
        if should_record:
            self.generation_best_records[self.iteration] = {
                "obj": best_obj,
                "code": population[best_sample_idx]["code"],
                "code_path": population[best_sample_idx]["code_path"],
            }

            # save the best code of current generation
            generation_best_code_path = (
                f"gen_{self.iteration}_best_code_{best_obj:.4f}.py"
            )
            with open(generation_best_code_path, "w") as f:
                f.write(population[best_sample_idx]["code"])

            # if wandb is enabled, upload the code
            if self.use_wandb:
                # upload the best code as an artifact to wandb
                gen_best_code_artifact = wandb.Artifact(
                    f"gen_{self.iteration}_best_solution", type="code"
                )
                gen_best_code_artifact.add_file(generation_best_code_path)
                wandb.log_artifact(gen_best_code_artifact)

            logging.info(
                f"Generation {self.iteration} best obj: {best_obj}, saved to {generation_best_code_path}"
            )

        # update the global best record
        if self.best_obj_overall is None or best_obj < self.best_obj_overall:
            self.best_obj_overall = best_obj
            self.best_code_overall = population[best_sample_idx]["code"]
            self.best_code_path_overall = population[best_sample_idx]["code_path"]

        # update the elitist individual
        if self.elitist is None or best_obj < self.elitist["obj"]:
            self.elitist = population[best_sample_idx]
            logger.info(f"Iteration {self.iteration}: Elitist: {self.elitist['obj']}")
            # run elitist code via test mode
            # take name, replace .txt with _test.txt
            stdout_filepath = self.elitist["stdout_filepath"].replace(".txt", "_test.txt")
            elitist_process = self._run_code(
                self.elitist,
                response_id="test",
                test=True,
                stdout_filepath=stdout_filepath,
            )
            # Wait until complete
            ade, fde = None, None
            try:
                logger.info("Testing elitist code...")
                elitist_process.communicate(
                    timeout=self.cfg.timeout
                )  # Wait for code execution to finish
                ade, fde = parse_metrics_from_stdout(stdout_filepath)
                logger.info(
                    f"Test results on dataset {self.cfg.problem.dataset}: ADE: {ade}, FDE: {fde}"
                )
            except subprocess.TimeoutExpired as e:
                logger.info(f"Elitist process timed out: {e}")
                elitist_process.kill()
            self.elitist["ade_test"] = ade
            self.elitist["fde_test"] = fde

        # record metrics
        if self.use_wandb:
            valid_count = sum(1 for ind in population if ind["exec_success"])
            valid_ratio = valid_count / len(population) if len(population) > 0 else 0

            unique_objs = list(set(objs))
            pop_diversity = np.std(objs) if len(objs) > 1 else 0

            metrics = {
                "iteration": self.iteration,
                "current_best_obj": best_obj,
                "avg_obj": np.mean(objs),
                "function_evals": self.function_evals,
                "valid_individuals": valid_count,
                "valid_ratio": valid_ratio,
                "population_diversity": pop_diversity,
                "unique_solutions": len(unique_objs),
                "history_records_count": len(self.generation_best_records),
                "current_mutation_rate": self.mutation_rate,
                "elitist/best_obj": self.best_obj_overall,
                "elitist/ade_test": self.elitist["ade_test"],
                "elitist/fde_test": self.elitist["fde_test"],
            }

            population_stats = {
                f"ind_{i}_obj": ind["obj"]
                for i, ind in enumerate(sorted(population, key=lambda x: x["obj"]))
                if i < 10 and ind["exec_success"]
            }
            metrics.update(population_stats)

            # record the best code of each generation to wandb
            wandb.log(metrics)

        logger.info(f"Iteration {self.iteration} finished...")

        # code --> response, .py --> .txt
        best_code_path_actual = self.best_code_path_overall.replace(
            ".py", ".txt"
        ).replace("code", "response")
        logger.info(
            f"Best obj: {self.best_obj_overall}, Best Code Path: {print_hyperlink(best_code_path_actual, self.best_code_path_overall)}"
        )
        logger.info(f"Function Evals: {self.function_evals}")
        self.iteration += 1

    def rank_select(self, population: list[dict]) -> list[dict]:
        """
        Rank-based selection, select individuals with probability proportional to their rank.
        """
        if self.problem_type == "black_box":
            population = [
                individual
                for individual in population
                if individual["exec_success"] and individual["obj"] < self.seed_ind["obj"]
            ]
        else:
            population = [
                individual for individual in population if individual["exec_success"]
            ]
        if len(population) < 2:
            return None
        # Sort population by objective value
        population = sorted(population, key=lambda x: x["obj"])
        ranks = [i for i in range(len(population))]
        probs = [1 / (rank + 1 + len(population)) for rank in ranks]
        # Normalize probabilities
        probs = [prob / sum(probs) for prob in probs]
        selected_population = []
        trial = 0
        while len(selected_population) < 2 * self.cfg.pop_size:
            trial += 1
            parents = np.random.choice(population, size=2, replace=False, p=probs)
            if parents[0]["obj"] != parents[1]["obj"]:
                selected_population.extend(parents)
            if trial > 1000:
                return None
        return selected_population

    def random_select(self, population: list[dict]) -> list[dict]:
        """
        Random selection, select individuals with equal probability.
        """
        selected_population = []
        # Eliminate invalid individuals
        if self.problem_type == "black_box":
            population = [
                individual
                for individual in population
                if individual["exec_success"] and individual["obj"] < self.seed_ind["obj"]
            ]
        else:
            population = [
                individual for individual in population if individual["exec_success"]
            ]

        # At least one valid individual is required (making it more tolerant)
        if len(population) < 1:
            logger.warning(
                f"Not enough valid individuals for selection: {len(population)}"
            )
            return None

        # If there is only one individual, duplicate it
        if len(population) == 1:
            logger.warning("Only one valid individual, duplicating it")
            population = population + population

        # If all objective values are identical, still perform selection
        if len(set([ind["obj"] for ind in population])) == 1:
            logger.warning("All individuals have identical objective values")
            selected = []
            for _ in range(2 * self.cfg.pop_size):
                selected.append(np.random.choice(population))
            return selected

        if len(population) % 2 != 0:
            logger.warning(
                f"Population length ({len(population)}) is not even. Adding a copy of the last individual."
            )
            population.append(population[-1])

        trial = 0
        max_trials = min(5000, 20 * self.cfg.pop_size * self.cfg.pop_size)  # more trials

        while len(selected_population) < 2 * self.cfg.pop_size and trial < max_trials:
            trial += 1
            replace_select = trial > max_trials // 2
            parents = np.random.choice(population, size=2, replace=replace_select)
            if parents[0]["obj"] != parents[1]["obj"] or trial > max_trials * 0.9:
                selected_population.extend(parents)

        if 0 < len(selected_population) < 2 * self.cfg.pop_size:
            logger.warning(f"Could only select {len(selected_population)} individuals")
            while len(selected_population) < 2 * self.cfg.pop_size:
                selected_population.append(np.random.choice(selected_population))

        if len(selected_population) == 0:
            logger.warning("Forcing selection despite no different-valued pairs")
            for _ in range(2 * self.cfg.pop_size):
                selected_population.append(np.random.choice(population))

        return selected_population

    def gen_short_term_reflection_prompt(
        self, ind1: dict, ind2: dict
    ) -> tuple[list[dict], str, str]:
        """
        Short-term reflection before crossovering two individuals.
        """
        if ind1["obj"] == ind2["obj"]:
            logger.warning(
                "Two individuals have the same objective value! Randomly selecting one as better."
            )
            better_ind, worse_ind = ind1, ind2
            if np.random.rand() < 0.5:
                better_ind, worse_ind = ind2, ind1
        else:
            if ind1["obj"] < ind2["obj"]:
                better_ind, worse_ind = ind1, ind2
            else:  # ind1["obj"] > ind2["obj"]
                better_ind, worse_ind = ind2, ind1

        worse_code = filter_code(worse_ind["code"])
        better_code = filter_code(better_ind["code"])

        system = self.system_reflector_prompt
        user = self.user_reflector_st_prompt.format(
            func_name=self.func_name,
            func_desc=self.func_desc,
            problem_desc=self.problem_desc,
            worse_code=worse_code,
            better_code=better_code,
            stats_info_worse=worse_ind["stats"],
            stats_info_better=better_ind["stats"],
        )
        message = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        # Print reflection prompt for the first iteration
        if self.print_short_term_reflection_prompt:
            logger.info(
                "Short-term Reflection Prompt: \nSystem Prompt: \n"
                + system
                + "\nUser Prompt: \n"
                + user
            )
            self.print_short_term_reflection_prompt = False
        return message, worse_code, better_code

    def short_term_reflection(
        self, population: list[dict]
    ) -> tuple[list[list[dict]], list[str], list[str]]:
        """
        Short-term reflection before crossovering two individuals.
        """
        messages_lst = []
        worse_code_lst = []
        better_code_lst = []
        for i in range(0, len(population), 2):
            # Select two individuals
            parent_1 = population[i]
            parent_2 = population[i + 1]

            # Short-term reflection
            messages, worse_code, better_code = self.gen_short_term_reflection_prompt(
                parent_1, parent_2
            )
            messages_lst.append(messages)
            worse_code_lst.append(worse_code)
            better_code_lst.append(better_code)

        # Asynchronously generate responses
        response_lst = self.short_reflector_llm.multi_chat_completion(messages_lst)
        return response_lst, worse_code_lst, better_code_lst

    def long_term_reflection(self, short_term_reflections: list[str]) -> None:
        """
        Long-term reflection before mutation.
        """
        system = self.system_reflector_prompt
        user = self.user_reflector_lt_prompt.format(
            problem_desc=self.problem_desc,
            prior_reflection=self.long_term_reflection_str,
            new_reflection="\n".join(short_term_reflections),
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        if self.print_long_term_reflection_prompt:
            logger.info(
                "Long-term Reflection Prompt: \nSystem Prompt: \n"
                + system
                + "\nUser Prompt: \n"
                + user
            )
            self.print_long_term_reflection_prompt = False

        self.long_term_reflection_str = self.long_reflector_llm.multi_chat_completion(
            [messages]
        )[0]

        # Write reflections to file
        file_name = f"problem_iter{self.iteration}_short_term_reflections.txt"
        with open(file_name, "w") as file:
            file.writelines("\n".join(short_term_reflections) + "\n")

        file_name = f"problem_iter{self.iteration}_long_term_reflection.txt"
        with open(file_name, "w") as file:
            file.writelines(self.long_term_reflection_str + "\n")

    def crossover(
        self, short_term_reflection_tuple: tuple[list[list[dict]], list[str], list[str]]
    ) -> list[dict]:
        reflection_content_lst, worse_code_lst, better_code_lst = (
            short_term_reflection_tuple
        )
        messages_lst = []
        for reflection, worse_code, better_code in zip(
            reflection_content_lst, worse_code_lst, better_code_lst
        ):
            # Crossover
            system = self.system_generator_prompt
            func_signature0 = self.func_signature.format(version=0)
            func_signature1 = self.func_signature.format(version=1)
            user = self.crossover_prompt.format(
                user_generator=self.user_generator_prompt,
                func_signature0=func_signature0,
                func_signature1=func_signature1,
                worse_code=worse_code,
                better_code=better_code,
                reflection=reflection,
                func_name=self.func_name,
            )
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
            messages_lst.append(messages)

            # Print crossover prompt for the first iteration
            if self.print_crossover_prompt:
                logger.info(
                    "Crossover Prompt: \nSystem Prompt: \n"
                    + system
                    + "\nUser Prompt: \n"
                    + user
                )
                self.print_crossover_prompt = False

        # Asynchronously generate responses
        response_lst = self.crossover_llm.multi_chat_completion(messages_lst)
        crossed_population = [
            self.response_to_individual(response, response_id)
            for response_id, response in enumerate(response_lst)
        ]

        assert len(crossed_population) == self.cfg.pop_size
        return crossed_population

    def softmax_sample_from_history(self, temperature=1.0):
        """
        Use softmax to sample a code from the history best records
        The temperature parameter controls the randomness of the sampling:
        - higher temperature (>1.0) increases randomness, sampling more uniformly from different generations
        - lower temperature (<1.0) makes the sampling more likely to choose better solutions
        """
        if not self.generation_best_records:
            # if there is no history records, use the current elitist
            logging.info("History records are empty, using the current elitist")
            return self.elitist["code"]

        # collect the scores and codes of all generations
        generations = list(self.generation_best_records.keys())
        # scores need to be converted to negative numbers (because we are minimizing the problem, smaller scores are better)
        scores = [-self.generation_best_records[gen]["obj"] for gen in generations]

        # apply softmax sampling
        # first normalize the scores to avoid numerical issues
        max_score = max(scores)
        exp_scores = [math.exp((score - max_score) / temperature) for score in scores]
        total = sum(exp_scores)
        probabilities = [exp_score / total for exp_score in exp_scores]

        # sample according to the probabilities
        selected_gen = np.random.choice(generations, p=probabilities)
        selected_code = self.generation_best_records[selected_gen]["code"]

        logging.info(
            f"Using softmax sampling to select the code of generation {selected_gen} (score: {self.generation_best_records[selected_gen]['obj']:.6f})"
        )

        return selected_code

    def mutate(self, mutation_rate: float = None) -> list[dict]:
        """Mutation operation, always use softmax to sample the base code from the history best records"""
        system = self.system_generator_prompt
        func_signature1 = self.func_signature.format(version=1)

        # use softmax sampling to select the base code
        # use a fixed temperature value, not based on the stagnation generation
        base_temperature = 1.0  # use a fixed temperature value
        base_code = self.softmax_sample_from_history(temperature=base_temperature)
        selected_gen = -1

        # find the corresponding generation
        for gen, record in self.generation_best_records.items():
            if record["code"] == base_code:
                selected_gen = gen
                break

        # record the information of the selected base code
        base_code_info = f"Generation {selected_gen}"

        logging.info(
            f"Using softmax to select the code of generation {selected_gen} as the base code"
        )

        user = self.mutation_prompt.format(
            user_generator=self.user_generator_prompt,
            reflection=self.long_term_reflection_str + self.external_knowledge,
            func_signature1=func_signature1,
            elitist_code=filter_code(base_code),  # use the selected code
            func_name=self.func_name,
            stats_info_elitist=self.elitist["stats"],
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        if self.print_mutate_prompt:
            logger.info(
                "Mutation Prompt: \nSystem Prompt: \n"
                + system
                + "\nUser Prompt: \n"
                + user
            )
            self.print_mutate_prompt = False
        mutation_rate = mutation_rate or self.mutation_rate
        responses = self.mutation_llm.multi_chat_completion(
            [messages], int(self.cfg.pop_size * mutation_rate)
        )
        population = [
            self.response_to_individual(response, response_id)
            for response_id, response in enumerate(responses)
        ]
        return population

    def evolve(self):
        while self.function_evals < self.cfg.max_fe:
            # verify if the population is valid
            need_regenerate = self.validate_and_fix_population()
            if need_regenerate:
                logger.warning("Population needs to be regenerated")
                # regenerate the population
                system = self.system_generator_prompt
                user = (
                    self.user_generator_prompt
                    + "\n"
                    + self.seed_prompt
                    + "\n"
                    + self.long_term_reflection_str
                    + "\n\n"
                    + "Previous population was invalid. Please generate diverse solutions."
                )
                messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ]

                responses = self.generator_llm.multi_chat_completion(
                    [messages],
                    self.cfg.pop_size,
                    temperature=self.generator_llm.temperature
                    + 0.4,  # use higher temperature for diversity
                )

                new_population = [
                    self.response_to_individual(
                        response,
                        response_id,
                        f"regenerate_iter{self.iteration}_ind{response_id}",
                    )
                    for response_id, response in enumerate(responses)
                ]

                # evaluate the new population
                new_population = self.evaluate_population(new_population)

                # keep the elitist if it is valid
                if self.elitist is not None and self.elitist["exec_success"]:
                    new_population.append(self.elitist)

                self.population = new_population

                # revalidate
                if self.validate_and_fix_population():
                    raise RuntimeError(
                        "Failed to regenerate a valid population after attempt"
                    )

                self.update_iter()
                continue

            # ensure the elitist is in the population
            population_to_select = (
                self.population
                if (self.elitist is None or self.elitist in self.population)
                else [self.elitist] + self.population
            )  # add elitist to population

            try:
                # modify to use random selection occasionally and elitist selection occasionally, increasing diversity
                selection_method = "random selection"
                if np.random.rand() < 0.7:  # 70% to use random selection
                    selected_population = self.random_select(population_to_select)
                else:  # 30% to use elitism selection
                    selected_population = self.rank_select(population_to_select)
                    selection_method = "elitism selection"
                    logging.info("This iteration uses elitism selection method")

                if selected_population is None:
                    logger.warning("Selection failed, trying to fix population")
                    # try to fix the population
                    if self.validate_and_fix_population():
                        # if need to regenerate
                        continue

                    selected_population = self.random_select(self.population)
                    if selected_population is None:
                        # final resolution
                        if self.elitist is not None and self.elitist["exec_success"]:
                            logger.warning("Emergency: creating selection from elitist")
                            selected_population = [self.elitist] * (2 * self.cfg.pop_size)
                        else:
                            raise RuntimeError(
                                "Selection failed after population recovery"
                            )
            except Exception as e:
                logger.error(f"Error during selection: {str(e)}")
                if self.validate_and_fix_population():
                    continue
                else:
                    if self.elitist is not None and self.elitist["exec_success"]:
                        logger.warning("Emergency: continuing with elitist only")
                        selected_population = [self.elitist] * (2 * self.cfg.pop_size)
                    else:
                        raise RuntimeError(
                            f"Selection failed with unrecoverable error: {str(e)}"
                        )

            try:
                short_term_reflection_tuple = self.short_term_reflection(
                    selected_population
                )  # (response_lst, worse_code_lst, better_code_lst)
            except Exception as e:
                logger.error(f"Error during short-term reflection: {str(e)}")
                if self.validate_and_fix_population():
                    continue
                else:
                    raise RuntimeError(
                        f"Short-term reflection failed with unrecoverable error: {str(e)}"
                    )

            crossed_population = self.crossover(short_term_reflection_tuple)

            self.population = self.evaluate_population(crossed_population)

            self.update_iter()

            self.long_term_reflection(
                [response for response in short_term_reflection_tuple[0]]
            )

            try:
                mutated_population = self.mutate(self.mutation_rate)

                self.population.extend(self.evaluate_population(mutated_population))
            except Exception as e:
                logger.error(f"Error during mutation: {str(e)}")
                pass

            self.update_iter()

        if self.use_wandb:
            wandb.log(
                {
                    "final_best_obj": self.best_obj_overall,
                    "total_iterations": self.iteration,
                    "total_function_evals": self.function_evals,
                }
            )

            best_code_artifact = wandb.Artifact("best_solution", type="code")
            best_code_path = f"best_solution_{self.best_obj_overall:.4f}.py"
            with open(best_code_path, "w") as f:
                f.write(self.best_code_overall)
            best_code_artifact.add_file(best_code_path)
            wandb.log_artifact(best_code_artifact)

            # save all generations best records to file
            all_generations_record_path = "all_generations_best_records.txt"
            with open(all_generations_record_path, "w") as f:
                f.write("Generation\tObjective\tCode Path\n")
                for gen, record in sorted(self.generation_best_records.items()):
                    f.write(f"{gen}\t{record['obj']:.6f}\t{record['code_path']}\n")

            # upload all generations records to wandb
            generations_record_artifact = wandb.Artifact(
                "generations_records", type="records"
            )
            generations_record_artifact.add_file(all_generations_record_path)
            wandb.log_artifact(generations_record_artifact)

            # Finish the run
            wandb.finish()

        return self.best_code_overall, self.best_code_path_overall

    def validate_and_fix_population(self):
        """
        Validate the current population and try to fix it
        Return whether to regenerate the population
        """
        if len(self.population) == 0:
            logger.warning("Empty population detected")
            return True

        valid_count = sum(1 for ind in self.population if ind["exec_success"])

        if valid_count == 0:
            logger.warning("No valid individuals in population")
            return True

        min_required = max(2, self.cfg.pop_size // 2)
        if valid_count < min_required:
            logger.warning(f"Too few valid individuals: {valid_count} < {min_required}")

            if self.elitist is not None and self.elitist["exec_success"]:
                logger.info(
                    "Attempting to recover by generating additional individuals through mutation"
                )

                additional_count = min(self.cfg.pop_size, 4)

                # use softmax to select multiple codes from history
                recovery_codes = []
                # use a higher temperature to increase diversity
                recovery_temperature = 2.0
                for _ in range(min(2, len(self.generation_best_records))):
                    code = self.softmax_sample_from_history(
                        temperature=recovery_temperature
                    )
                    if code not in recovery_codes:
                        recovery_codes.append(code)

                # ensure at least one code
                if not recovery_codes:
                    recovery_codes = [self.elitist["code"]]

                recovery_populations = []

                # generate mutation versions for each selected code
                for i, base_code in enumerate(recovery_codes):
                    # using elitist code for generation
                    system = self.system_generator_prompt
                    user = (
                        self.mutation_prompt.format(
                            user_generator=self.user_generator_prompt,
                            reflection=self.long_term_reflection_str
                            + self.external_knowledge,
                            func_signature1=self.func_signature.format(version=1),
                            elitist_code=filter_code(base_code),
                            func_name=self.func_name,
                        )
                        + "\n\nPlease generate mutation versions that are significantly different from the base code, to increase exploration diversity."
                    )

                    messages = [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ]

                    # assign additional generations based on the number of selected codes
                    current_additional_count = additional_count // len(recovery_codes)
                    if i == 0:
                        current_additional_count += additional_count % len(recovery_codes)

                    responses = self.mutation_llm.multi_chat_completion(
                        [messages],
                        current_additional_count,
                        temperature=self.mutation_llm.temperature + 0.4,
                    )

                    current_recovery_population = [
                        self.response_to_individual(
                            response,
                            len(self.population)
                            + len(recovery_populations)
                            + response_id,
                            f"recovery_iter{self.iteration}_base{i}_ind{response_id}",
                        )
                        for response_id, response in enumerate(responses)
                    ]

                    recovery_populations.extend(current_recovery_population)

                recovery_population = self.evaluate_population(recovery_populations)
                self.population.extend(recovery_population)

                valid_count = sum(1 for ind in self.population if ind["exec_success"])
                if valid_count >= min_required:
                    logger.info(
                        f"Population recovery successful: now have {valid_count} valid individuals"
                    )
                    return False

            return True

        return False
