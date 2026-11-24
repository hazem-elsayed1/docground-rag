from src.hybrid import fuse


def test_fusion_of_partial_overlapping_and_duplicate_rankings():
    a, b, c = ({"id": key, "score": 4} for key in "abc")
    results = fuse({"s": [b, b, a], "d": [c, a]}, 3)
    assert [hit["id"] for hit in results] == ["a", "b", "c"]
    assert results[0]["score"] == 2 / 62
    assert results[0]["components"] == {key: {"score": 4, "rank": 2} for key in "sd"}
    assert fuse({"s": [], "d": [c]}, 10)[0]["id"] == "c"
    assert fuse({}, 1) == []
