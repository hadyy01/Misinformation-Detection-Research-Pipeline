from false_information_flagging.preprocessing import clean_text


def test_clean_text_removes_html_urls_and_stopwords():
    value = "<p>This is a sample article</p> https://example.com [ref]"
    cleaned = clean_text(
        value,
        lowercase=True,
        strip_html=True,
        strip_urls=True,
        remove_bracketed_text=True,
        remove_stopwords=True,
    )
    assert "https" not in cleaned
    assert "sample" in cleaned
    assert "this" not in cleaned

