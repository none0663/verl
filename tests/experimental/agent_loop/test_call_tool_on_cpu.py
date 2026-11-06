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

"""Unit tests for ToolAgentLoop._call_tool error handling (no GPU required).

Tests that malformed tool calls return specific, actionable error messages
instead of generic exception strings.
"""

import unittest
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import datasets
import pytest
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from verl.tools.schemas import ToolResponse
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn


@dataclass
class FakeFunctionCall:
    """Minimal FunctionCall for testing."""

    name: str
    arguments: str


@dataclass
class FakeAgentData:
    """Minimal AgentData for testing."""

    tools_kwargs: dict = field(default_factory=dict)


class FakeTool:
    """A fake tool that succeeds."""

    def __init__(self, name: str):
        self.name = name

    async def create(self, create_kwargs=None):
        return "instance_1", ToolResponse()

    async def execute(self, instance_id, parameters, **kwargs):
        return ToolResponse(text=f"OK: {parameters}"), 1.0, {}

    async def release(self, instance_id):
        pass


class FakeFailingTool(FakeTool):
    """A fake tool that raises during execute."""

    async def execute(self, instance_id, parameters, **kwargs):
        raise RuntimeError("database connection failed")


class FakeLongResponseTool(FakeTool):
    """A fake tool that returns a long response."""

    def __init__(self, name: str, text: str):
        super().__init__(name)
        self.text = text

    async def execute(self, instance_id, parameters, **kwargs):
        return ToolResponse(text=self.text), 1.0, {}


def _make_tool_agent_loop(
    tools: dict[str, Any],
    max_tool_response_length: int = 10000,
    tool_response_truncate_side: str = "left",
):
    """Create a minimal ToolAgentLoop instance with only the fields _call_tool needs."""
    from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop

    mock = MagicMock(spec=ToolAgentLoop)
    mock.tools = tools
    mock.max_tool_response_length = max_tool_response_length
    mock.tool_response_truncate_side = tool_response_truncate_side
    # Bind the real _call_tool method to our mock
    mock._call_tool = ToolAgentLoop._call_tool.__get__(mock, ToolAgentLoop)
    return mock


class TestCallToolErrorHandling(unittest.IsolatedAsyncioTestCase):
    """Test ToolAgentLoop._call_tool error handling for malformed tool calls."""

    def setUp(self):
        self.tools = {
            "calculator": FakeTool("calculator"),
            "search": FakeTool("search"),
        }
        self.loop = _make_tool_agent_loop(self.tools)
        self.agent_data = FakeAgentData()

    async def test_valid_tool_call(self):
        """Valid tool call should succeed."""
        tool_call = FakeFunctionCall(name="calculator", arguments='{"a": 3, "b": 5}')
        response, reward, _ = await self.loop._call_tool(tool_call, {}, self.agent_data)
        assert reward == 1.0
        assert "OK" in response.text

    async def test_unknown_function_name(self):
        """Unknown function name should list available tools."""
        tool_call = FakeFunctionCall(name="calculater", arguments='{"a": 3}')
        response, reward, _ = await self.loop._call_tool(tool_call, {}, self.agent_data)
        assert reward == 0.0
        assert "Unknown function" in response.text
        assert "calculater" in response.text
        assert "calculator" in response.text
        assert "search" in response.text

    async def test_invalid_json_arguments(self):
        """Invalid JSON arguments should report parse error."""
        tool_call = FakeFunctionCall(name="calculator", arguments="{a: 3}")
        response, reward, _ = await self.loop._call_tool(tool_call, {}, self.agent_data)
        assert reward == 0.0
        assert "Invalid JSON" in response.text
        assert "calculator" in response.text

    async def test_empty_arguments(self):
        """Empty string arguments should report parse error."""
        tool_call = FakeFunctionCall(name="calculator", arguments="")
        response, reward, _ = await self.loop._call_tool(tool_call, {}, self.agent_data)
        assert reward == 0.0
        assert "Invalid JSON" in response.text

    async def test_none_arguments(self):
        """None arguments should report error."""
        tool_call = FakeFunctionCall(name="calculator", arguments=None)
        response, reward, _ = await self.loop._call_tool(tool_call, {}, self.agent_data)
        assert reward == 0.0
        assert "Invalid JSON" in response.text

    async def test_tool_execution_error(self):
        """Tool execution failure should include tool name in error."""
        tools = {"failing_tool": FakeFailingTool("failing_tool")}
        loop = _make_tool_agent_loop(tools)
        tool_call = FakeFunctionCall(name="failing_tool", arguments='{"query": "test"}')
        response, reward, _ = await loop._call_tool(tool_call, {}, self.agent_data)
        assert reward == 0.0
        assert "failing_tool" in response.text
        assert "database connection failed" in response.text

    async def test_left_truncation_keeps_response_tail(self):
        """Left truncation should drop the left side and preserve the response tail."""
        tool_response = (
            "Search results for capital of France:\n"
            "1. Lyon is a major city with a long Roman history.\n"
            "2. Marseille is a large port city in southern France.\n"
            "3. The final retrieved snippet says the capital is Paris.\n"
            "Final answer: Paris"
        )
        tools = {"search": FakeLongResponseTool("search", tool_response)}
        loop = _make_tool_agent_loop(tools, max_tool_response_length=19, tool_response_truncate_side="left")
        tool_call = FakeFunctionCall(name="search", arguments="{}")
        response, reward, _ = await loop._call_tool(tool_call, {}, self.agent_data)
        assert reward == 1.0
        assert response.text.startswith("(truncated)...")
        assert response.text.endswith("Final answer: Paris")

    async def test_right_truncation_keeps_response_head(self):
        """Right truncation should drop the right side and preserve the response head."""
        tool_response = (
            "Search results for capital of France:\n"
            "1. Lyon is a major city with a long Roman history.\n"
            "2. Marseille is a large port city in southern France.\n"
            "3. The final retrieved snippet says the capital is Paris.\n"
            "Final answer: Paris"
        )
        tools = {"search": FakeLongResponseTool("search", tool_response)}
        loop = _make_tool_agent_loop(tools, max_tool_response_length=19, tool_response_truncate_side="right")
        tool_call = FakeFunctionCall(name="search", arguments="{}")
        response, reward, _ = await loop._call_tool(tool_call, {}, self.agent_data)
        assert reward == 1.0
        assert response.text.startswith("Search results")
        assert response.text.endswith("...(truncated)")
        assert "Final answer: Paris" not in response.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "extra_info",
    [
        {"index": 0},
        {"index": 0, "tools_kwargs": {"search": {"create_kwargs": {"limit": 1}}}},
        {"index": 0, "tools_kwargs": {"calculator": {"execute_kwargs": {"precision": 2}}}},
    ],
    ids=["missing-tools-kwargs", "missing-tool", "missing-create-kwargs"],
)
async def test_tool_call_with_kwargs_from_mixed_parquet(tmp_path, extra_info):
    """Schema alignment must not turn omitted tool configuration into a failed call."""
    configured_kwargs = {"ground_truth": "42"}
    configured_extra_info = {
        "index": 1,
        "tools_kwargs": {"calculator": {"create_kwargs": configured_kwargs}},
    }
    data_files = []
    for index, info in enumerate([extra_info, configured_extra_info]):
        path = tmp_path / f"samples-{index}.parquet"
        datasets.Dataset.from_list(
            [{"prompt": [{"role": "user", "content": "Calculate 3 + 5."}], "extra_info": info}]
        ).to_parquet(path)
        data_files.append(str(path))

    dataset = RLHFDataset(
        data_files=data_files,
        tokenizer=None,
        config=OmegaConf.create({"filter_overlong_prompts": False, "cache_dir": str(tmp_path / "cache")}),
    )
    batch = next(iter(DataLoader(dataset, batch_size=2, collate_fn=collate_fn)))
    # Hugging Face fills the omitted struct field with None at one of these levels.
    missing_kwargs = batch["tools_kwargs"][0]
    assert (
        missing_kwargs is None
        or missing_kwargs["calculator"] is None
        or missing_kwargs["calculator"]["create_kwargs"] is None
    )

    tool = FakeTool("calculator")
    tool.create = AsyncMock(wraps=tool.create)
    tool.execute = AsyncMock(wraps=tool.execute)
    tool.release = AsyncMock(wraps=tool.release)
    loop = _make_tool_agent_loop({"calculator": tool})
    tool_call = FakeFunctionCall(name="calculator", arguments='{"a": 3, "b": 5}')

    for tools_kwargs, expected_kwargs in zip(batch["tools_kwargs"], [{}, configured_kwargs], strict=True):
        agent_data = FakeAgentData(tools_kwargs=tools_kwargs)
        response, reward, _ = await loop._call_tool(tool_call, tools_kwargs, agent_data)
        assert response.text == "OK: {'a': 3, 'b': 5}"
        assert reward == 1.0
        tool.create.assert_awaited_with(create_kwargs=expected_kwargs)
        tool.execute.assert_awaited_with("instance_1", {"a": 3, "b": 5}, agent_data=agent_data)
        tool.release.assert_awaited_with("instance_1")

    assert tool.create.await_count == tool.execute.await_count == tool.release.await_count == 2


if __name__ == "__main__":
    unittest.main()
