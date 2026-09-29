"""Reflex-based instinct-dual-4b recipe: CPU tests, no model weights.

The GOLDEN table was produced by running Reflex's own prompt code
(https://github.com/kshetrajna12/reflex at b77f03bab8e00d7319e64550e062c1eac8111875,
``git archive b77f03b src/reflex``) as the production service does:
``build_branches(qid, question, PromptFormat(chat=True, no_think=True, style="markdown"), 2)``
with the default seed 0 and ``PromptFormat.prefix(state)``, for the two requests
below. ``ids_len`` / ``ids_sha256`` are the token ids of ``encode(prefix) +
encode(branch)`` (add_special_tokens=False) under the Qwen3.5-4B tokenizer at
revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a; they are only checked when
that tokenizer is present in the local Hugging Face cache.
"""

import os
import hashlib
import json
from pathlib import Path

import pytest

from instinct.recipes.base import Encoded
from instinct.recipes.reflex_dual_4b import RECIPE

GOLDEN = json.loads(r"""
{
 "example": {
  "request": {
   "state": {
    "customer": "My package says delivered but it is not here.",
    "order_status": "delivered",
    "days_since_delivery": 1
   },
   "questions": {
    "intent": {
     "type": "choice",
     "instructions": "What does the customer want?",
     "criteria": {
      "track_order": "Find out where the order is",
      "cancel_order": "Cancel the order",
      "refund": "Get money back",
      "other": "Something else"
     }
    },
    "escalate": {
     "type": "noul",
     "instructions": "The customer is asking to speak to a human agent."
    },
    "urgency": {
     "type": "score",
     "instructions": "How urgent is this request?",
     "criteria": [
      "Not urgent",
      "Somewhat urgent",
      "Urgent",
      "Critical"
     ]
    }
   }
  },
  "prefix": "<|im_start|>system\nYou are a System One decision model. You read the State and answer each Question by choosing exactly one of the listed options. You never explain. You answer with the single option label only.<|im_end|>\n<|im_start|>user\n# Evidence\n{\n  \"customer\": \"My package says delivered but it is not here.\",\n  \"order_status\": \"delivered\",\n  \"days_since_delivery\": 1\n}\n\n",
  "questions": {
   "intent": [
    {
     "text": "# Criterion\nWhat does the customer want?\n\n# Options\nA. track_order: Find out where the order is\nB. cancel_order: Cancel the order\nC. refund: Get money back\nD. other: Something else\n\nRespond with only the letter of the best option.\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
     "labels": [
      "A",
      "B",
      "C",
      "D"
     ],
     "keys": [
      "track_order",
      "cancel_order",
      "refund",
      "other"
     ],
     "ids_len": 157,
     "ids_sha256": "ab1e3f14d5c1e0557dd7bc63e5795326eef32bf379ed1da079e5e60e3ee38452"
    },
    {
     "text": "# Criterion\nWhat does the customer want?\n\n# Options\nA. other: Something else\nB. track_order: Find out where the order is\nC. refund: Get money back\nD. cancel_order: Cancel the order\n\nRespond with only the letter of the best option.\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
     "labels": [
      "A",
      "B",
      "C",
      "D"
     ],
     "keys": [
      "other",
      "track_order",
      "refund",
      "cancel_order"
     ],
     "ids_len": 157,
     "ids_sha256": "fb77efec1ae06ac0f2d7d3500da0cd4ebb164fa650407ed5d06c702fc4c158de"
    }
   ],
   "escalate": [
    {
     "text": "# Criterion\nThe customer is asking to speak to a human agent.\n\n# Options\nA. yes: The statement is true.\nB. no: The statement is false.\n\nRespond with only the letter of the best option.\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
     "labels": [
      "A",
      "B"
     ],
     "keys": [
      true,
      false
     ],
     "ids_len": 146,
     "ids_sha256": "741e32e432933edee474869883b735849ac12253fdcf339f6012de1861f1e373"
    },
    {
     "text": "# Criterion\nThe customer is asking to speak to a human agent.\n\n# Options\nA. no: The statement is false.\nB. yes: The statement is true.\n\nRespond with only the letter of the best option.\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
     "labels": [
      "A",
      "B"
     ],
     "keys": [
      false,
      true
     ],
     "ids_len": 146,
     "ids_sha256": "19458b715960ccbecf0576c2cd85a0129276f4461954f81c0875b94444e789f2"
    }
   ],
   "urgency": [
    {
     "text": "# Criterion\nHow urgent is this request?\n\n# Options\nA. (level 0 of 3) Not urgent\nB. (level 1 of 3) Somewhat urgent\nC. (level 2 of 3) Urgent\nD. (level 3 of 3) Critical\n\nRespond with only the letter of the level that best matches.\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
     "labels": [
      "A",
      "B",
      "C",
      "D"
     ],
     "keys": [
      0,
      1,
      2,
      3
     ],
     "ids_len": 175,
     "ids_sha256": "c17e762d4d14479ac6f064986a81462c4acb9ce964153370f19f654daeaf11dc"
    },
    {
     "text": "# Criterion\nHow urgent is this request?\n\n# Options\nA. (level 0 of 3) Not urgent\nB. (level 3 of 3) Critical\nC. (level 2 of 3) Urgent\nD. (level 1 of 3) Somewhat urgent\n\nRespond with only the letter of the level that best matches.\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
     "labels": [
      "A",
      "B",
      "C",
      "D"
     ],
     "keys": [
      0,
      3,
      2,
      1
     ],
     "ids_len": 175,
     "ids_sha256": "91aad959b0c5bd7524f66597bfb0efc4b03faf33c07dd2b5e1b76e059988678b"
    }
   ]
  }
 },
 "text_state": {
  "request": {
   "state": "Ticket #41: the printer on floor 3 jams every morning.",
   "questions": {
    "q-choice": {
     "type": "choice",
     "instructions": "Which team owns this?",
     "criteria": {
      "it": "IT support",
      "fac": "Facilities",
      "hr": "Human resources",
      "sec": "Security",
      "fin": "Finance"
     }
    },
    "q-noul": {
     "type": "noul",
     "instructions": "The problem is recurring.",
     "criteria": {
      "true": "It happens repeatedly.",
      "false": "It is a one-off."
     }
    },
    "q-score": {
     "type": "score",
     "instructions": "How severe is it?",
     "criteria": [
      "Cosmetic",
      "Minor",
      "Moderate",
      "Major",
      "Blocking"
     ]
    }
   }
  },
  "prefix": "<|im_start|>system\nYou are a System One decision model. You read the State and answer each Question by choosing exactly one of the listed options. You never explain. You answer with the single option label only.<|im_end|>\n<|im_start|>user\n# Evidence\nTicket #41: the printer on floor 3 jams every morning.\n\n",
  "questions": {
   "q-choice": [
    {
     "text": "# Criterion\nWhich team owns this?\n\n# Options\nA. it: IT support\nB. fac: Facilities\nC. hr: Human resources\nD. sec: Security\nE. fin: Finance\n\nRespond with only the letter of the best option.\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
     "labels": [
      "A",
      "B",
      "C",
      "D",
      "E"
     ],
     "keys": [
      "it",
      "fac",
      "hr",
      "sec",
      "fin"
     ],
     "ids_len": 129,
     "ids_sha256": "821a7911e1c08d69bd42ed65a67344d48eb80f60d543b258d2a77ad48f3e06d1"
    },
    {
     "text": "# Criterion\nWhich team owns this?\n\n# Options\nA. sec: Security\nB. fin: Finance\nC. fac: Facilities\nD. hr: Human resources\nE. it: IT support\n\nRespond with only the letter of the best option.\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
     "labels": [
      "A",
      "B",
      "C",
      "D",
      "E"
     ],
     "keys": [
      "sec",
      "fin",
      "fac",
      "hr",
      "it"
     ],
     "ids_len": 129,
     "ids_sha256": "7a3de1b9b676855c8a7166f860e222d58fd68bd50e5aa257259ac0c6beb126ee"
    }
   ],
   "q-noul": [
    {
     "text": "# Criterion\nThe problem is recurring.\n\n# Options\nA. yes: It happens repeatedly.\nB. no: It is a one-off.\n\nRespond with only the letter of the best option.\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
     "labels": [
      "A",
      "B"
     ],
     "keys": [
      true,
      false
     ],
     "ids_len": 117,
     "ids_sha256": "f0bd613a9149531cec1e7892f782e723ddea1c9abb89bfcf556a54fb2c1909ff"
    },
    {
     "text": "# Criterion\nThe problem is recurring.\n\n# Options\nA. no: It is a one-off.\nB. yes: It happens repeatedly.\n\nRespond with only the letter of the best option.\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
     "labels": [
      "A",
      "B"
     ],
     "keys": [
      false,
      true
     ],
     "ids_len": 117,
     "ids_sha256": "5e98ecde2d9c2b9a4fd3dafd70725c986df332fb6f684d67f2b187b45597572b"
    }
   ],
   "q-score": [
    {
     "text": "# Criterion\nHow severe is it?\n\n# Options\nA. (level 0 of 4) Cosmetic\nB. (level 1 of 4) Minor\nC. (level 2 of 4) Moderate\nD. (level 3 of 4) Major\nE. (level 4 of 4) Blocking\n\nRespond with only the letter of the level that best matches.\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
     "labels": [
      "A",
      "B",
      "C",
      "D",
      "E"
     ],
     "keys": [
      0,
      1,
      2,
      3,
      4
     ],
     "ids_len": 159,
     "ids_sha256": "1ba88508e3119e171e5f6d9e700c0455a0a073e53db48fd8ad67a34ac999b3ff"
    },
    {
     "text": "# Criterion\nHow severe is it?\n\n# Options\nA. (level 4 of 4) Blocking\nB. (level 1 of 4) Minor\nC. (level 0 of 4) Cosmetic\nD. (level 2 of 4) Moderate\nE. (level 3 of 4) Major\n\nRespond with only the letter of the level that best matches.\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
     "labels": [
      "A",
      "B",
      "C",
      "D",
      "E"
     ],
     "keys": [
      4,
      1,
      0,
      2,
      3
     ],
     "ids_len": 159,
     "ids_sha256": "2c667394560530b9a19831748b738b8e8fb58ba979c7f824110f9dbc20b63df1"
    }
   ]
  }
 }
}""")

def _hub_cache():
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]) / "hub"
    return Path.home() / ".cache/huggingface/hub"


SNAPSHOT = (
    _hub_cache() / "models--Qwen--Qwen3.5-4B/snapshots" / "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
)


class CharTokenizer:
    """One token per character; prefix and branch are encoded separately, as in production."""

    def encode(self, text, add_special_tokens=True):
        assert add_special_tokens is False
        return [ord(c) for c in text]


def decode(ids):
    return "".join(chr(i) for i in ids)


def candidate(kind, key):
    return ("yes" if key else "no") if kind == "noul" else str(key)


def cases():
    for name, gold in GOLDEN.items():
        records = {qid: (kind, record) for qid, kind, record in RECIPE.records(gold["request"])}
        for qid, branches in gold["questions"].items():
            yield name, gold, qid, branches, *records[qid]


CASES = list(cases())
IDS = [f"{c[0]}-{c[2]}" for c in CASES]


def test_recipe_identity():
    assert RECIPE.name == "instinct-dual-4b"
    assert RECIPE.hf_repo == "Qwen/Qwen3.5-4B"
    assert RECIPE.revision == "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    assert RECIPE.temperature == 1.0 and RECIPE.readout == "full_head"
    assert RECIPE.max_length == 262144


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_prompt_text_and_orders_match_reflex_goldens(case):
    _, gold, qid, branches, kind, record = case
    orders = RECIPE.orders(record)
    assert len(orders) == 2 and orders[0] != orders[1]
    assert orders == [[candidate(kind, k) for k in b["keys"]] for b in branches]
    for order, expected in zip(orders, branches):
        encoded = RECIPE.encode(CharTokenizer(), record, order)
        assert isinstance(encoded, Encoded) and encoded.readout_index == -1
        assert decode(encoded.input_ids) == gold["prefix"] + expected["text"]
        assert encoded.label_token_ids == [ord(label) for label in expected["labels"]]
        assert encoded.candidates == order


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_first_order_is_given_and_second_is_seeded_shuffle_or_swap(case):
    _, _, _, _, kind, record = case
    given = [c["id"] for c in record["criteria"]]
    first, second = RECIPE.orders(record)
    assert first == given and sorted(second) == sorted(given) and second != first
    if len(given) == 2:
        assert second == given[::-1]


def test_orders_depend_on_question_id_only():
    body = {
        "state": "s",
        "questions": {
            "a": {"type": "choice", "instructions": "i", "criteria": {k: k * 2 for k in "wxyz"}},
            "b": {"type": "choice", "instructions": "i", "criteria": {k: k * 2 for k in "wxyz"}},
        },
    }
    by_id = {qid: RECIPE.orders(record)[1] for qid, _, record in RECIPE.records(body)}
    alone = RECIPE.records({"state": "s", "questions": {"b": body["questions"]["b"]}})
    assert RECIPE.orders(alone[0][2])[1] == by_id["b"]


def test_records_stay_in_generic_form_and_keep_raw_state():
    for name, gold in GOLDEN.items():
        for qid, kind, record in RECIPE.records(gold["request"]):
            assert record["id"] == qid and record["group_id"] == "systemone"
            assert record["primitive"] == ("noul" if kind == "noul" else "choice")
            assert record["reflex"]["state"] == gold["request"]["state"]
            if kind == "noul":
                assert [c["id"] for c in record["criteria"]] == ["yes", "no"]
            if kind == "score":
                assert [c["id"] for c in record["criteria"]] == [
                    str(i) for i in range(len(gold["request"]["questions"][qid]["criteria"]))
                ]


def test_object_state_is_pretty_printed_like_reflex():
    _, gold, qid, branches, kind, record = next(c for c in CASES if c[0] == "example")
    assert '# Evidence\n{\n  "customer": "My package' in gold["prefix"]
    assert record["state"].startswith('{"customer"')  # generic compact form is not what is sent


def test_noul_uses_supplied_wording_and_lettered_yes_no():
    _, _, _, branches, _, record = next(c for c in CASES if c[2] == "q-noul")
    assert "A. yes: It happens repeatedly.\nB. no: It is a one-off." in branches[0]["text"]
    assert "A. no: It is a one-off.\nB. yes: It happens repeatedly." in branches[1]["text"]


def test_label_tokens_are_bare_letters_and_prompt_has_empty_think_prefill():
    _, gold, qid, branches, kind, record = CASES[0]
    encoded = RECIPE.encode(CharTokenizer(), record, RECIPE.orders(record)[0])
    text = decode(encoded.input_ids)
    assert text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert gold["prefix"].startswith("<|im_start|>system\nYou are a System One decision model.")
    assert encoded.label_token_ids[0] == ord("A")


def test_merge_is_temperature_one_softmax_average_in_caller_ids():
    _, _, _, _, kind, record = next(c for c in CASES if c[2] == "q-noul")
    branches = [RECIPE.encode(CharTokenizer(), record, o) for o in RECIPE.orders(record)]
    # branch 1 (A=yes, B=no): 2:0 logits; branch 2 (A=no, B=yes): 0:2 logits => yes twice
    merged = RECIPE.merge(record, branches, [[2.0, 0.0], [0.0, 2.0]])
    p = 1 / (1 + 2.718281828459045 ** -2)
    assert merged == pytest.approx([p, 1 - p])  # order of record["criteria"]: yes, no


def test_branch_and_total_limits():
    class Long(CharTokenizer):
        def encode(self, text, add_special_tokens=True):
            return [1] * (5000 if text.startswith("# Criterion") else 3)

    _, _, _, _, kind, record = CASES[0]
    with pytest.raises(ValueError, match="too long"):
        RECIPE.encode(Long(), record, RECIPE.orders(record)[0])


def test_rejects_orders_that_are_not_the_reflex_pair():
    _, _, _, _, kind, record = next(c for c in CASES if c[2] == "q-choice")
    given = RECIPE.orders(record)[0]
    with pytest.raises(ValueError, match="option orders"):
        RECIPE.encode(CharTokenizer(), record, sorted(given))


def test_score_levels_capped_at_reflex_limit():
    body = {"state": "s", "questions": {"q": {"type": "score", "instructions": "i",
                                              "criteria": [f"l{i}" for i in range(11)]}}}
    with pytest.raises(ValueError, match="at most 10"):
        RECIPE.records(body)


def test_plain_generic_records_are_supported():
    record = {"id": "x", "group_id": "g", "state": "s", "instructions": "Pick.",
              "primitive": "choice",
              "criteria": [{"id": "a", "description": "First"}, {"id": "b", "description": "Second"}]}
    assert RECIPE.orders(record) == [["a", "b"], ["b", "a"]]
    encoded = RECIPE.encode(CharTokenizer(), record, ["b", "a"])
    assert "A. b: Second\nB. a: First" in decode(encoded.input_ids)


@pytest.mark.skipif(not SNAPSHOT.is_dir(), reason="Qwen3.5-4B tokenizer not in the local HF cache")
@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_token_ids_match_reference_tokenizer(case):
    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True)
    _, _, _, branches, kind, record = case
    for order, expected in zip(RECIPE.orders(record), branches):
        encoded = RECIPE.encode(tokenizer, record, order)
        assert len(encoded.input_ids) == expected["ids_len"]
        digest = hashlib.sha256(json.dumps(encoded.input_ids).encode()).hexdigest()
        assert digest == expected["ids_sha256"]
        assert encoded.label_token_ids == [tokenizer.encode(l, add_special_tokens=False)[0]
                                           for l in expected["labels"]]
