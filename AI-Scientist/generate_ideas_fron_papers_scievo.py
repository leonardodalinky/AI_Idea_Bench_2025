import codecs
import json
import os
import os.path as osp
import sys

from loguru import logger
from tqdm import tqdm

sys.path.append(osp.join(osp.dirname(__file__), ".."))

from extract_one_paper_conten import get_one_paper_conten
from LLM.Deepseek_v3 import Deepseek
from prompt_template.process_one_paper import get_one_paper_input

if p := os.getenv("SCIEVO_DIR"):
    sys.path.insert(0, p)
else:
    raise ImportError(
        "SCIEVO_DIR environment variable not set. Please set it to the root directory of SciEvo."
    )

from bench_workflows.register_models.gemini import (
    register_gemini3_medium_high_models,
    register_gemini_low_medium_models,
    register_gemini_medium_high_models,
)
from bench_workflows.register_models.gpt import (
    register_gpt_low_medium_models,
    register_gpt_medium_high_models,
)
from scievo.workflows.ideation_workflow import IdeationWorkflow, run_ideation_workflow


def append_to_json_file(file_path, new_data, index):
    if os.path.exists(file_path):
        with codecs.open(file_path, "r", encoding="utf-8") as file:
            existing_data = json.load(file)
            file.close()
    else:
        existing_data = []

    existing_data.append({"index": index, "model_result": new_data})
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(existing_data, file, ensure_ascii=False, indent=4)
    print("Data appended successfully.")


def data_check(file_path, new_index):
    if os.path.exists(file_path):
        with codecs.open(file_path, "r", encoding="utf-8") as file:
            existing_data = json.load(file)
            file.close()
    else:
        return False

    existing_indexes = set(item["index"] for item in existing_data)

    return new_index in existing_indexes


def ensure_folder_exists(json_file_path):

    folder_path = os.path.dirname(json_file_path)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


if __name__ == "__main__":
    import warnings

    from bs4.builder import XMLParsedAsHTMLWarning

    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

    # minimum log level to INFO
    logger.remove()
    logger.add(sys.stdout, level="INFO")

    api_key_deepseek = os.getenv("OPENAI_API_KEY")
    assert api_key_deepseek is not None, "OPENAI_API_KEY environment variable not set."

    assert os.getenv("S2_API_KEY") is not None, "S2_API_KEY environment variable not set."

    # NOTE (kelin): we use GPT instead
    model_api = Deepseek([api_key_deepseek], None, "gpt-5-nano")

    # Register models based on choice
    MODEL = "gemini3-medium-high"
    logger.info(f"Registering models: {MODEL}")
    match MODEL:
        case "gpt-low-medium":
            register_gpt_low_medium_models()
        case "gpt-medium-high":
            register_gpt_medium_high_models()
        case "gemini-low-medium":
            register_gemini_low_medium_models()
        case "gemini-medium-high":
            register_gemini_medium_high_models()
        case "gemini3-medium-high":
            register_gemini3_medium_high_models()

    #######################################################################################################################################

    find_cite_result_directory = "./target_paper_data.json"  # target_paper_data path

    with open(find_cite_result_directory, "r") as f:
        datasets = json.load(f)
        f.close()

    final_ideas_save_path = "./model_output/AI-Scientist/final_ideas.json"  # model_output path
    cited_paper_conten_save_path = "./dataset_temple/cited_paper_conten.json"  # dataset_temple path

    # NOTE: you can change them depend on your need
    num_ideas = 2

    ensure_folder_exists(final_ideas_save_path)
    ensure_folder_exists(cited_paper_conten_save_path)

    for input_data in tqdm(datasets):
        if data_check(final_ideas_save_path, input_data["index"]):
            continue

        if input_data["summary"]["revised_topic"]:
            topic = input_data["summary"]["revised_topic"]
        else:
            topic = input_data["summary"]["topic"]

        cite_paper_id = 0
        all_paper_input = ""
        all_paper_conten = []
        for cites_paper in tqdm(input_data["find_cite"]["top_references"]):
            paper_local_path = cites_paper["paper_local_path"]
            try:
                cite_paper_conten = get_one_paper_conten(model_api, paper_local_path, topic)
            except Exception as e:
                logger.warning("Error processing paper {}: {}", paper_local_path, e)
                continue
            idea = cite_paper_conten["idea"]
            experiment = cite_paper_conten["experiment"]
            entities = cite_paper_conten["entities"]
            one_paper_input = get_one_paper_input(cite_paper_id, idea, entities, experiment)
            all_paper_conten.append(
                {"paper_path": paper_local_path, "model_result": one_paper_input}
            )
            all_paper_input = all_paper_input + "\n" + one_paper_input

        if data_check(cited_paper_conten_save_path, input_data["index"]):
            pass
        else:
            append_to_json_file(cited_paper_conten_save_path, all_paper_conten, input_data["index"])

        # Generate
        try:
            workflow: IdeationWorkflow = run_ideation_workflow(
                research_domain=topic,
                user_query=f"Generate research ideas based on the the research topic and the following papers: \n{all_paper_input}",
                workspace_path="./tmp_workspace",
            )
            research_ideas = workflow.research_ideas
            novelty_accessments = workflow.idea_novelty_assessments
            assert len(research_ideas) > 0, "No ideas generated."
            assert len(novelty_accessments) > 0, "No novelty assessments generated."
        except Exception as e:
            logger.warning(
                "Error during ideation workflow for index {}: {}", input_data["index"], e
            )
            continue
        novelty_accessments.sort(key=lambda x: x["novelty_score"], reverse=True)
        selected = novelty_accessments[:num_ideas]
        selected_ideas = [research_ideas[a["idea_idx"]] for a in selected]

        novel_ideas = []

        # In <JSON>, provide the new idea in JSON format with the following fields:
        # - "Name": A shortened descriptor of the idea. Lowercase, no spaces, underscores allowed.
        # - "Title": A title for the idea, will be used for the report writing.
        # - "Motivation":Provide a background for your idea, summarizing relevant past work. Identify shortcomings in previous research and highlight the specific problems that remain unsolved and that you aim to address.
        # - "Experiment": An outline of the implementation. E.g. which functions need to be added or modified, how results will be obtained, ...
        # - "Interestingness": A rating from 1 to 10 (lowest to highest).
        # - "Feasibility": A rating from 1 to 10 (lowest to highest).
        # - "Novelty": A rating from 1 to 10 (lowest to highest).
        for idea, accessment in zip(selected_ideas, selected):
            novel_ideas.append(
                {
                    "Name": idea["title"],
                    "Title": idea["title"],
                    "Motivation": idea["rationale"],
                    "Experiment": idea["experiment"],
                    "Interestingness": 5,  # fixed
                    "Feasibility": 5,  # fixed
                    "Novelty": accessment["novelty_score"],
                }
            )

        append_to_json_file(final_ideas_save_path, novel_ideas, input_data["index"])
