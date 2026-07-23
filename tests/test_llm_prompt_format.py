from app.services.llm_service import (
    MARKDOWN_OUTPUT_INSTRUCTIONS,
    LLMService,
)


def test_default_instructions_include_shared_markdown_contract():
    instructions = LLMService()._build_instructions("Bạn là trợ lý trả lời câu hỏi.")

    assert MARKDOWN_OUTPUT_INSTRUCTIONS in instructions
    assert "Bảng bắt buộc có hàng tiêu đề" in instructions


def test_internal_plain_text_instructions_can_opt_out_of_markdown_contract():
    instructions = LLMService()._build_instructions(
        "Chỉ trả về JSON.",
        include_markdown_instructions=False,
    )

    assert MARKDOWN_OUTPUT_INSTRUCTIONS not in instructions


def test_rag_prompt_contains_shared_markdown_contract():
    prompt = LLMService().build_prompt(question="Tóm tắt tài liệu")

    assert MARKDOWN_OUTPUT_INSTRUCTIONS in prompt
