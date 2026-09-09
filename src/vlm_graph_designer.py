"""VLM-based Graph Designer: extract partial scene graphs from text and/or images."""
import ast
import base64
import os
import re
from typing import List, Optional, Tuple

from openai import OpenAI


def build_graph_designer_prompt(
    object_types: List[str],
    predicate_types: List[str],
    user_text: str = "",
) -> str:
    obj_list = ", ".join(f"{i}:'{n}'" for i, n in enumerate(object_types))
    pred_list = ", ".join(f"{i}:'{n}'" for i, n in enumerate(predicate_types))
    return f"""You are an experienced 3D bedroom designer. Extract main objects and spatial relationships from the user input (text and/or image).

Available object classes: [{obj_list}]
Available spatial relations: [{pred_list}]

User text: {user_text or "(none)"}

Respond with exactly three lines:
object list: [0:class_index, 1:class_index, ...]
spatial relationship triplets: [[subject_idx, relation_idx, object_idx], ...]
description: a concise natural-language scene description

Rules:
- Each triplet [s, p, o] means object s has relation p relative to object o.
- Use inverse relations for the opposite direction when needed.
- Only use object and relation indices from the lists above.
- If the input is a top-view diagram, map above/below to behind/in front of accordingly."""


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def call_vlm(
    prompt: str,
    image_path: Optional[str] = None,
    model: str = "gpt-4o",
) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("Set OPENAI_API_KEY before running VLM Graph Designer.")

    client = OpenAI(
        api_key=api_key,
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )

    content = [{"type": "text", "text": prompt}]
    if image_path:
        ext = os.path.splitext(image_path)[1].lower().lstrip(".")
        mime = "jpeg" if ext in ("jpg", "jpeg") else ext
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/{mime};base64,{encode_image(image_path)}",
            },
        })

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an experienced 3D room designer."},
            {"role": "user", "content": content},
        ],
    )
    return response.choices[0].message.content.strip()


def parse_vlm_response(response: str) -> Tuple[List[int], List[List[int]], str]:
    lines = [ln.strip() for ln in response.strip().split("\n") if ln.strip()]
    if len(lines) < 3:
        raise ValueError(f"VLM response must contain 3 lines, got:\n{response}")

    object_list_line = lines[0]
    object_list_coarse = re.search(r"^[^:]*:(.*)$", object_list_line).group(1).split(",")
    object_list = [int(re.search(r":(\d+)\D*$", x).group(1)) for x in object_list_coarse]

    triplet_list_line = lines[1]
    triplet_list_coarse = re.search(r"^[^:]*:(.*)$", triplet_list_line).group(1).strip()
    triplet_list = ast.literal_eval(triplet_list_coarse)

    description_line = lines[2]
    description = description_line.split(":", 1)[1].strip().strip('"')
    return object_list, triplet_list, description


def design_graph(
    object_types: List[str],
    predicate_types: List[str],
    user_text: str = "",
    image_path: Optional[str] = None,
    model: str = "gpt-4o",
) -> Tuple[List[int], List[List[int]], str]:
    prompt = build_graph_designer_prompt(object_types, predicate_types, user_text)
    response = call_vlm(prompt, image_path=image_path, model=model)
    return parse_vlm_response(response)
