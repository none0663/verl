# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader


def test_rl_collate_fn():
    from verl.utils.dataset.rl_dataset import collate_fn

    max_prompt_length = 5

    test_data = [
        {
            # test tensor
            "input_ids": torch.randint(0, 10, (max_prompt_length,)),
            # test fixed length (1) list within a batch
            "messages": [{"role": "user", "content": "Hi."}],
            # test variable length list within a batch
            "raw_prompt_ids": [1, 2, 3, 4],
            # test string
            "ability": "math",
            # test dict
            "reward_model": {"ground_truth": 5, "style": "rule"},
            # test empty dict
            "tools_kwargs": {},
        },
        {
            "input_ids": torch.randint(0, 10, (max_prompt_length,)),
            "messages": [{"role": "user", "content": "Hello."}],
            "raw_prompt_ids": [1, 2, 3],
            "ability": "toolcall",
            "reward_model": {
                "ground_truth": '[{"name": "rgb_to_cmyk", "arguments": {"r": 0, "g": 0, "b": 255}}]',
                "style": "rule",
            },
            "tools_kwargs": {},
        },
    ]

    batch_size = len(test_data)
    batch = collate_fn(test_data)

    # Tensor part
    assert batch["input_ids"].shape == (batch_size, max_prompt_length)
    assert isinstance(batch["input_ids"], torch.Tensor)

    # Non-tensor parts
    expected_types = {
        "messages": list,
        "raw_prompt_ids": list,
        "ability": str,
        "reward_model": dict,
        "tools_kwargs": dict,
    }

    for key, dtype in expected_types.items():
        assert batch[key].shape == (batch_size,), (
            f"Expected shape {(batch_size,)} for '{key}', but got {batch[key].shape}"
        )
        assert isinstance(batch[key][0], dtype), (
            f"'{key}' should contain elements of type {dtype}, but got {type(batch[key][0])}"
        )


@pytest.mark.parametrize("field", ["input_ids", "extra_info"])
@pytest.mark.parametrize("missing_row", [0, 1])
def test_rl_collate_fn_rejects_missing_fields(field, missing_row):
    from verl.utils.dataset.rl_dataset import collate_fn

    rows = [
        {"input_ids": torch.tensor([1, 2]), "extra_info": {"index": 0}},
        {"input_ids": torch.tensor([3, 4]), "extra_info": {"index": 1}},
    ]
    del rows[missing_row][field]

    with pytest.raises(ValueError, match=rf"sample 1.*{field}"):
        collate_fn(rows)


@pytest.mark.parametrize("non_tensor", [None, 2, [2], np.array([2])])
@pytest.mark.parametrize("tensor_first", [False, True])
def test_rl_collate_fn_rejects_mixed_tensor_fields(non_tensor, tensor_first):
    from verl.utils.dataset.rl_dataset import collate_fn

    values = [torch.tensor([1]), non_tensor]
    if not tensor_first:
        values.reverse()

    with pytest.raises(TypeError, match="input_ids.*sample 1"):
        collate_fn([{"input_ids": value} for value in values])


@pytest.mark.parametrize("rows", [[], [{}, {}]])
def test_rl_collate_fn_empty(rows):
    from verl.utils.dataset.rl_dataset import collate_fn

    assert collate_fn(rows) == {}


def test_rl_collate_fn_preserves_rows_in_dataloader():
    from verl.utils.dataset.rl_dataset import collate_fn

    rows = [
        {"input_ids": torch.tensor([1, 2]), "extra_info": None, "raw_prompt_ids": [1, 2]},
        {"raw_prompt_ids": [3], "extra_info": {"index": 1}, "input_ids": torch.tensor([3, 4])},
        {"input_ids": torch.tensor([5, 6]), "extra_info": None, "raw_prompt_ids": []},
    ]
    batches = list(DataLoader(rows, batch_size=2, collate_fn=collate_fn))

    assert len(batches) == 2
    for batch, expected_rows in zip(batches, [rows[:2], rows[2:]], strict=True):
        torch.testing.assert_close(batch["input_ids"], torch.stack([row["input_ids"] for row in expected_rows]))
        for field in ["extra_info", "raw_prompt_ids"]:
            assert batch[field].shape == (len(expected_rows),)
            assert batch[field].dtype == object
            assert batch[field].tolist() == [row[field] for row in expected_rows]


def test_rl_collate_fn_reports_invalid_dataloader_batch():
    from verl.utils.dataset.rl_dataset import collate_fn

    loader = DataLoader([{"input_ids": torch.tensor([1])}, {"input_ids": None}], batch_size=2, collate_fn=collate_fn)

    with pytest.raises(TypeError, match="input_ids.*sample 1"):
        next(iter(loader))
