from app.utils.markdown_tables import normalize_markdown_tables


def test_expands_collapsed_markdown_table_rows():
    value = "| Mục | Nội dung | |---|---| | Mức lương | 18.500.000 VND/tháng | | Phụ cấp | 2.500.000 VND/tháng |"

    assert normalize_markdown_tables(value) == (
        "| Mục | Nội dung |\n"
        "| --- | --- |\n"
        "| Mức lương | 18.500.000 VND/tháng |\n"
        "| Phụ cấp | 2.500.000 VND/tháng |"
    )


def test_preserves_normal_markdown_table():
    value = "| Mục | Nội dung |\n|---|---|\n| Mức lương | 18.500.000 VND/tháng |"

    assert normalize_markdown_tables(value) == value


def test_preserves_prose_with_pipes():
    value = "Điều kiện A | Điều kiện B"

    assert normalize_markdown_tables(value) == value
