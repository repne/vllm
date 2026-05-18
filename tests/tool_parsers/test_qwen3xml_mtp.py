"""
Test Qwen3 XML tool parser MTP streaming issues:
1. Complete tool call delivered in single MTP batch
2. Multiple complete tool calls in single MTP batch
3. Tool call followed by trailing text in single MTP batch
4. Deferral edge case: </parameter> at end of buffer
"""
import json
from unittest.mock import MagicMock

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
)
from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.tool_parsers.qwen3xml_tool_parser import (
    Qwen3XMLToolParser,
    StreamingXMLToolCallParser,
)


def make_request(tools=None):
    """Create a minimal ChatCompletionRequest."""
    request = MagicMock(spec=ChatCompletionRequest)
    request.tools = tools
    request.tool_choice = "auto"
    request.n = 1
    return request


def make_tools():
    """Create a simple tool definition for testing."""
    return [
        ChatCompletionToolsParam(
            type="function",
            function={
                "name": "get_weather",
                "description": "Get the weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The location to get weather for",
                        },
                    },
                    "required": ["location"],
                },
            },
        ),
    ]


def make_parser(tools=None):
    """Create a Qwen3XMLToolParser instance."""
    tokenizer = MagicMock()
    tokenizer.get_vocab = MagicMock(return_value={})
    return Qwen3XMLToolParser(tokenizer=tokenizer, tools=tools or make_tools())


class TestMTPStreamingQwen3XML:
    """Test MTP streaming scenarios with Qwen3 XML parser."""

    def test_complete_tool_call_single_batch(self):
        """
        Test that a complete tool call delivered in a single MTP batch
        is correctly parsed and emitted as structured tool calls,
        not as raw XML content.
        """
        parser = make_parser()
        request = make_request(tools=make_tools())

        # Reset for clean test
        parser.parser.reset_streaming_state()
        parser.parser._streaming_mode = True
        parser.parser.set_tools(make_tools())
        parser.prev_tool_call_arr = []
        parser.streamed_args_for_tool = []

        # Feed complete tool call in one shot (MTP scenario)
        complete_tc = (
            "<tool_call>"
            "<function=get_weather>"
            "\n"
            "<parameter=location>"
            "\nSan Francisco"
            "</parameter>"
            "\n</function>"
            "\n_"
        )

        result = parser.parser.parse_single_streaming_chunks(complete_tc)

        # The result should contain tool_calls, not raw XML content
        assert result.tool_calls is not None, (
            "Tool call should be parsed, not returned as raw XML content"
        )
        assert len(result.tool_calls) > 0, "Should have at least one tool call"

        # Find the tool call with the function name
        func_tc = None
        for tc in result.tool_calls:
            if tc.function and tc.function.name:
                func_tc = tc
                break
        assert func_tc is not None, "Should have a tool call with a function name"
        assert func_tc.function.name == "get_weather"

    def test_streaming_mtp_batch_after_partial(self):
        """
        Test that when the first batch brings us partway through a tool call
        and the MTP batch completes it, the parsing still works correctly.
        """
        parser = make_parser()
        request = make_request(tools=make_tools())

        parser.parser.reset_streaming_state()
        parser.parser._streaming_mode = True
        parser.parser.set_tools(make_tools())
        parser.prev_tool_call_arr = []
        parser.streamed_args_for_tool = []

        # Feed the beginning of a tool call (up to parameter value, no end)
        initial_text = (
            "<tool_call><function=get_weather>\n<parameter=location>\nSan Francisco"
        )
        delta1 = parser.parser.parse_single_streaming_chunks(initial_text)

        # At this point, function name and param name should be parsed

        # Now MTP delivers the rest: </parameter>\n</function>\n_
        mtp_batch = "\n</parameter>\n</function>\n_"
        delta2 = parser.parser.parse_single_streaming_chunks(mtp_batch)

        assert delta2 is not None, "Should return a delta for the MTP batch"

        # Verify no raw XML leaked into content
        content = delta2.content if delta2 else None
        if content:
            assert "</parameter>" not in content, (
                f"Raw XML tag leaked into content: {content}"
            )
            assert "</function>" not in content, (
                f"Raw XML tag leaked into content: {content}"
            )

    def test_prev_tool_call_arr_updated(self):
        """
        Test that prev_tool_call_arr is properly updated with function name
        and complete arguments after a complete tool call is streamed.
        This is needed for the serving layer to detect tool_calls finish_reason
        and compute remaining args.
        """
        parser = make_parser()
        request = make_request(tools=make_tools())

        parser.parser.reset_streaming_state()
        parser.parser._streaming_mode = True
        parser.parser.set_tools(make_tools())
        parser.prev_tool_call_arr = []
        parser.streamed_args_for_tool = []

        complete_tc = (
            "<tool_call><function=get_weather>"
            "\n<parameter=location>"
            "\nSan Francisco"
            "</parameter>"
            "\n</function>"
            "\n_"
        )

        delta = parser.extract_tool_calls_streaming(
            previous_text="",
            current_text=complete_tc,
            delta_text=complete_tc,
            previous_token_ids=[],
            current_token_ids=[1, 2, 3, 4, 5],
            delta_token_ids=[1, 2, 3, 4, 5],
            request=request,
        )

        # prev_tool_call_arr should have the function name and arguments
        assert len(parser.prev_tool_call_arr) > 0, (
            "prev_tool_call_arr should have entries after parsing tool call"
        )
        entry = parser.prev_tool_call_arr[0]
        assert entry["name"] == "get_weather", (
            f"Function name should be 'get_weather', got '{entry['name']}'"
        )
        assert entry["arguments"] is not None and entry["arguments"] != "", (
            f"Arguments should not be empty, got '{entry['arguments']}'"
        )
        assert "location" in entry["arguments"] or "San Francisco" in entry[
            "arguments"
        ], (
            f"Arguments should contain 'location' or 'San Francisco', "
            f"got '{entry['arguments']}'"
        )

    def test_streamed_args_for_tool_updated(self):
        """
        Test that streamed_args_for_tool is properly updated so the serving
        layer's remaining args check works correctly.
        """
        parser = make_parser()
        request = make_request(tools=make_tools())

        parser.parser.reset_streaming_state()
        parser.parser._streaming_mode = True
        parser.parser.set_tools(make_tools())
        parser.prev_tool_call_arr = []
        parser.streamed_args_for_tool = []

        complete_tc = (
            "<tool_call><function=get_weather>"
            "\n<parameter=location>"
            "\nSan Francisco"
            "</parameter>"
            "\n</function>"
            "\n_"
        )

        delta = parser.extract_tool_calls_streaming(
            previous_text="",
            current_text=complete_tc,
            delta_text=complete_tc,
            previous_token_ids=[],
            current_token_ids=[1, 2, 3, 4, 5],
            delta_token_ids=[1, 2, 3, 4, 5],
            request=request,
        )

        assert len(parser.streamed_args_for_tool) > 0, (
            "streamed_args_for_tool should have entries"
        )
        args = parser.streamed_args_for_tool[0]
        assert args is not None and len(args) > 0, (
            f"streamed_args_for_tool should not be empty, got '{args}'"
        )


class TestMultipleToolCallsMTP:
    """Test multiple tool calls delivered in a single MTP batch."""

    def test_two_tool_calls_single_batch(self):
        """
        Test that two complete tool calls delivered in a single MTP batch
        are both correctly parsed.
        """
        parser = make_parser()
        request = make_request(tools=make_tools())

        parser.parser.reset_streaming_state()
        parser.parser._streaming_mode = True
        parser.parser.set_tools(make_tools())
        parser.prev_tool_call_arr = []
        parser.streamed_args_for_tool = []

        two_tc = (
            "<tool_call><function=get_weather>"
            "\n<parameter=location>"
            "\nSan Francisco</parameter>"
            "\n</function>"
            "\n_"
            "<tool_call><function=get_weather>"
            "\n<parameter=location>"
            "\nNew York</parameter>"
            "\n</function>"
            "\n_"
        )

        delta = parser.extract_tool_calls_streaming(
            previous_text="",
            current_text=two_tc,
            delta_text=two_tc,
            previous_token_ids=[],
            current_token_ids=[1, 2, 3, 4, 5],
            delta_token_ids=[1, 2, 3, 4, 5],
            request=request,
        )

        # Both tool calls should be tracked
        assert len(parser.prev_tool_call_arr) >= 2, (
            f"Should track at least two tool calls, "
            f"got {len(parser.prev_tool_call_arr)}"
        )


class TestRawParserStreaming:
    """Test the raw StreamingXMLToolCallParser directly."""

    @pytest.fixture
    def raw_parser(self):
        """Create a raw StreamingXMLToolCallParser."""
        p = StreamingXMLToolCallParser()
        p.set_tools(make_tools())
        return p

    def test_parameter_end_deferred_without_lookahead(self, raw_parser):
        """
        Test that </parameter> at the very end of the buffer with no
        trailing content is deferred (not processed).
        """
        text = (
            "<tool_call><function=get_weather>"
            "\n<parameter=location>"
            "\nSan Francisco</parameter>"
        )
        raw_parser._streaming_mode = True
        raw_parser.streaming_buffer += text
        result = raw_parser._process_complete_xml_elements()

        # The parameter end should be deferred because nothing follows it
        assert raw_parser.current_param_name == "location", (
            "Parameter should still be open (deferred due to no lookahead)"
        )

    def test_parameter_end_processed_with_lookahead(self, raw_parser):
        """
        Test that </parameter> with content after it in the buffer IS
        processed (MTP scenario).
        """
        text = (
            "<tool_call><function=get_weather>"
            "\n<parameter=location>"
            "\nSan Francisco</parameter>"
            "\n</function>"
            "\n_"
        )
        raw_parser._streaming_mode = True
        raw_parser.streaming_buffer += text
        result = raw_parser._process_complete_xml_elements()

        assert result, "Should find complete elements"
        # The parameter should have been closed (lookahead available)
        assert raw_parser.current_param_name is None, (
            "Parameter should be closed after processing with lookahead"
        )

    def test_tool_call_complete_in_single_batch(self, raw_parser):
        """
        Test that a complete tool call in a single batch is fully parsed
        with all deltas emitted.
        """
        text = (
            "<tool_call><function=get_weather>"
            "\n<parameter=location>"
            "\nSan Francisco</parameter>"
            "\n</function>"
            "\n_"
        )
        raw_parser._streaming_mode = True

        result = raw_parser.parse_single_streaming_chunks(text)

        # Should have deltas emitted
        assert len(raw_parser.deltas) > 0, "Should have emitted deltas"

        # Check for function name delta
        has_func_name = False
        has_args = False
        for d in raw_parser.deltas:
            if d.tool_calls:
                for tc in d.tool_calls:
                    if tc.function:
                        if tc.function.name:
                            has_func_name = True
                        if tc.function.arguments:
                            has_args = True

        assert has_func_name, "Should have emitted function name delta"
        assert has_args, "Should have emitted arguments delta"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
