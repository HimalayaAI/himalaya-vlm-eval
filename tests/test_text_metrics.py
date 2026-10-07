import unicodedata

import pytest

from himalaya_vlm_eval import metrics as M
from himalaya_vlm_eval import text as T


def test_nfc_makes_decomposed_and_precomposed_nukta_equal():
    precomposed = "क़"  # क़
    decomposed = "क़"  # क + ़
    assert precomposed != decomposed
    assert T.canonical(precomposed) == T.canonical(decomposed)
    assert M.cer(precomposed, [decomposed]) == 0.0
    assert M.exact_match(decomposed + "लम", [precomposed + "लम"]) == 1.0


def test_canonical_collapses_unicode_whitespace():
    assert T.canonical("नेपाल सरकार   छ\n") == "नेपाल सरकार छ"


def test_loose_strips_danda_punctuation_zero_width_and_case():
    assert T.loose("धन्यवाद।") == T.loose("धन्यवाद")
    assert T.loose("श्लोक॥") == "श्लोक"
    assert T.loose("Hello, World!") == "hello world"
    assert T.loose("क्‍ष") == T.loose("क्ष")
    assert M.loose_match("धन्यवाद ।", ["धन्यवाद"]) == 1.0
    assert M.exact_match("धन्यवाद ।", ["धन्यवाद"]) == 0.0


def test_canonical_keeps_danda_for_cer():
    # CER compares what was printed; a dropped danda is an error.
    assert M.cer("धन्यवाद", ["धन्यवाद।"]) == pytest.approx(1 / len("धन्यवाद।"))


def test_clean_model_output():
    assert T.clean_model_output("<think>hmm</think>\nनेपाल") == "नेपाल"
    assert T.clean_model_output("```text\nनेपाल\n```") == "नेपाल"
    assert T.clean_model_output("नेपाल<|endoftext|>") == "नेपाल"
    assert T.clean_model_output("a ``` b") == "a ``` b"


@pytest.mark.parametrize(
    "pred,ref,expected",
    [("abc", "abc", 0.0), ("abd", "abc", 1 / 3), ("", "abc", 1.0), ("abcdef", "abc", 1.0),
     ("x", "", 1.0), ("", "", 0.0)],
)
def test_cer(pred, ref, expected):
    assert M.cer(pred, [ref]) == pytest.approx(expected)


def test_wer_and_ned_and_char_accuracy():
    assert M.wer("नेपाल सरकार", ["नेपाल सरकारको"]) == 0.5
    assert 0 <= M.ned("abcdefgh", ["abc"]) <= 1
    assert M.char_accuracy("zzzzzzzz", ["ab"]) == 0.0


def test_edit_distance_matches_reference_implementation():
    def ref(a, b):
        dp = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            prev, dp[0] = dp[0], i
            for j, cb in enumerate(b, 1):
                prev, dp[j] = dp[j], min(dp[j] + 1, dp[j - 1] + 1, prev + (ca != cb))
        return dp[-1]

    for a, b in [("kitten", "sitting"), ("नेपाल", "नेपल"), ("", "abc"), ("ab", "ba")]:
        assert M.edit_distance(a, b) == ref(a, b)


def test_anls():
    assert M.anls("Paris", ["paris"]) == 1.0
    assert M.anls("Pariss", ["paris"]) == pytest.approx(1 - 1 / 6)
    assert M.anls("London", ["paris"]) == 0.0  # below threshold
    assert M.anls("b", ["a", "b"]) == 1.0


def test_relaxed_accuracy():
    assert M.relaxed_accuracy("104", ["100"]) == 1.0
    assert M.relaxed_accuracy("106", ["100"]) == 0.0
    assert M.relaxed_accuracy("12%", ["12"]) == 1.0
    assert M.relaxed_accuracy("1,000", ["1000"]) == 1.0
    assert M.relaxed_accuracy("Yes", ["yes"]) == 1.0
    assert M.relaxed_accuracy("0", ["0"]) == 1.0


def test_contains_match():
    assert M.contains_match("The answer is Kathmandu.", ["kathmandu"]) == 1.0
    assert M.contains_match("Pokhara", ["kathmandu"]) == 0.0


def test_every_metric_has_a_worst_value():
    assert set(M.METRICS) == set(M.WORST)
    with pytest.raises(KeyError):
        M.get_metric("nope")


def test_bootstrap_ci_is_deterministic_and_brackets_the_mean():
    vals = [0.0] * 50 + [1.0] * 50
    lo, hi = M.bootstrap_ci(vals, seed=1)
    assert (lo, hi) == M.bootstrap_ci(vals, seed=1)
    assert lo < 0.5 < hi
    assert hi - lo < 0.3
    assert M.bootstrap_ci([0.3]) == (0.3, 0.3)
    with pytest.raises(ValueError):
        M.bootstrap_ci([])


def test_nfc_is_idempotent_on_generated_ground_truth():
    gt = "यस्तो बेलामा सत्ताको स्वाद पाएका"
    assert unicodedata.is_normalized("NFC", gt)
    assert T.canonical(gt) == gt


# --- akshara CER, loose WER, multi-reference (issues #6, #7, #8) -------------------------


def test_aksharas_split_like_a_reader_sees_characters():
    assert T.aksharas("किताब") == ["कि", "ता", "ब"]
    assert T.aksharas("विद्यालय") == ["वि", "द्या", "ल", "य"]
    assert T.aksharas("क्षत्रिय") == ["क्ष", "त्रि", "य"]
    assert T.aksharas("नेपाल।") == ["ने", "पा", "ल", "।"]
    assert T.aksharas("अं ४२") == ["अं", " ", "४", "२"]


def test_acer_counts_a_dropped_vowel_sign_as_one_character():
    assert M.cer("कताब", ["किताब"]) == pytest.approx(0.2)
    assert M.acer("कताब", ["किताब"]) == pytest.approx(1 / 3)
    assert M.acer("विदयालय", ["विद्यालय"]) == pytest.approx(0.5)
    assert M.acer("किताब", ["किताब"]) == 0.0


def test_acer_ignores_zero_width_characters():
    assert M.acer("क्‌ष", ["क्ष"]) == 0.0
    assert M.acer("क्‍ष", ["क्ष"]) == 0.0
    assert M.cer("क्‌ष", ["क्ष"]) > 0  # the code-point metric is unchanged


def test_acer_empty_cases():
    assert M.acer("", [""]) == 0.0
    assert M.acer("क", [""]) == 1.0


def test_wer_loose_ignores_danda_and_punctuation():
    assert M.wer("नेपाल सरकारले बजेट ल्यायो", ["नेपाल सरकारले बजेट ल्यायो।"]) == 0.25
    assert M.wer_loose("नेपाल सरकारले बजेट ल्यायो", ["नेपाल सरकारले बजेट ल्यायो।"]) == 0.0
    assert M.wer_loose("नेपाल सरकार छ", ["नेपाल सरकार हो"]) == pytest.approx(1 / 3)


def test_transcription_metrics_take_the_best_reference():
    refs = ["नेपाल सरकार", "नेपाल सरकारले"]
    pred = "नेपाल सरकारले"
    assert M.cer(pred, refs) == 0.0
    assert M.wer(pred, refs) == 0.0
    assert M.ned(pred, refs) == 0.0
    assert M.acer(pred, refs) == 0.0
    assert M.char_accuracy(pred, refs) == 1.0


def test_single_reference_results_are_unchanged():
    assert M.cer("abd", ["abc"]) == pytest.approx(1 / 3)
    assert M.wer("नेपाल सरकार", ["नेपाल सरकारको"]) == 0.5
    assert M.cer("x", []) == 1.0


def test_new_metrics_are_registered_with_worst_case():
    for name in ("acer", "wer_loose"):
        assert M.get_metric(name) is getattr(M, name)
        assert M.WORST[name] == 1.0
