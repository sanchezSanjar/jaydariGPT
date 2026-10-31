"""Prompt formatting shared by training and inference."""

# Every section header starts with this; generation stops once the model emits it.
STOP_STRING = "###"


def format_instruction(instruction: str, input_text: str = "", response: str = "") -> str:
    instruction = (instruction or "").strip()
    input_text = (input_text or "").strip()
    response = (response or "").strip()

    if input_text:
        return (
            "### Instruction:\n"
            f"{instruction}\n\n"
            "### Input:\n"
            f"{input_text}\n\n"
            "### Response:\n"
            f"{response}"
        )
    return (
        "### Instruction:\n"
        f"{instruction}\n\n"
        "### Response:\n"
        f"{response}"
    )


def format_chat_prompt(instruction: str) -> str:
    return format_instruction(instruction, response="").rstrip() 
