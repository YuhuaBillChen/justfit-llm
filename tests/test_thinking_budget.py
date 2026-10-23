"""Hard-budget validation, wire bytes, and native speculative boundary behavior."""
from dataclasses import replace
import importlib.util
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'source'))
from server import protocol as wire


class ThinkingBudget(unittest.TestCase):
    def frame(self, **changes):
        return wire.RequestFrame(request_id=1, priority=wire.RequestPriority.NORMAL,
            absolute_deadline_unix_micros=100, remaining_deadline_micros=100,
            logical_max_output_tokens=128, prompt_tokens=(1, 2), sampling=wire.SamplingParameters(),
            seed=0, constraint=wire.ConstraintMode.NONE, image_spans=(), image_pixels=b'',
            score_tokens=(), generation_prompt_tokens=0, flags=wire.RequestFlag(0), **changes)

    def test_budget_is_on_wire_and_protocol_is_versioned(self):
        frame = self.frame(thinking_budget_tokens=31, thinking_end_token_id=99)
        data = wire.serialize_message(frame)
        self.assertEqual(wire.PROTOCOL_VERSION, 9)
        self.assertEqual(struct.unpack_from('<II', data, 16+86), (31, 99))
        self.assertEqual(struct.unpack_from('<II', data, 16+wire.REQUEST_FIXED_BYTES), (1, 2))

    def test_wire_rejects_invalid_or_inapplicable_budget(self):
        valid = self.frame(thinking_budget_tokens=31, thinking_end_token_id=99)
        for fields in [dict(thinking_budget_tokens=True), dict(thinking_budget_tokens=-1),
                       dict(thinking_budget_tokens=127), dict(thinking_end_token_id=0),
                       dict(thinking_budget_tokens=0),
                       dict(logical_max_output_tokens=0, score_tokens=(1, 2))]:
            with self.subTest(fields=fields), self.assertRaises(wire.ProtocolError):
                wire.serialize_message(replace(valid, **fields))

    def test_responses_and_messages_keep_the_budget(self):
        from server.api_shapes import (responses_to_chat_body, canonical_responses_input,
                                       anthropic_to_chat_body)
        response, _ = responses_to_chat_body(dict(model='test', thinking_budget=8,
            reasoning=dict(effort='medium')), canonical_responses_input('hello'))
        self.assertEqual(response['thinking_budget'], 8)
        message, _ = anthropic_to_chat_body(dict(model='test', max_tokens=128,
            messages=[dict(role='user', content='hello')],
            thinking=dict(type='enabled', budget_tokens=8)), thinking_resolver=lambda value: value)
        self.assertEqual(message['thinking_budget'], 8)

    def test_frontend_rejects_budget_without_thinking_or_answer_room(self):
        from server.thinking_budget import prepare_thinking_budget
        from server.errors import APIError
        tokenizer = types.SimpleNamespace(convert_tokens_to_ids=lambda text: 99)
        self.assertEqual(prepare_thinking_budget(31, tokenizer, True, 128), (31, 99))
        self.assertEqual(prepare_thinking_budget(None, tokenizer, False, 128), (0, 0))
        for value, enabled, output in [(True, True, 128), (-1, True, 128), (1.5, True, 128),
                                       ('31', True, 128), (31, False, 128), (31, True, 32)]:
            with self.subTest(value=value, enabled=enabled, output=output), self.assertRaises(APIError):
                prepare_thinking_budget(value, tokenizer, enabled, output)

    def test_native_cap_prevents_speculative_overshoot_and_preserves_natural_close(self):
        compiler = shutil.which('c++')
        self.assertIsNotNone(compiler, 'a C++20 compiler is required for native boundary tests')
        with tempfile.TemporaryDirectory() as temporary:
            binary = Path(temporary)/'thinking-budget-test'
            subprocess.run([compiler, '-std=c++20', '-Wall', '-Wextra', '-Werror',
                '-I'+str(ROOT/'source/runtime'), str(ROOT/'tests/thinking_budget_state.cpp'),
                '-o', str(binary)], check=True, capture_output=True)
            subprocess.run([str(binary)], check=True)

    def test_python_frames_decode_in_actual_native_protocol(self):
        compiler = shutil.which('c++')
        self.assertIsNotNone(compiler)
        with tempfile.TemporaryDirectory() as temporary:
            binary = Path(temporary)/'wire-test'
            subprocess.run([compiler, '-std=c++20', '-I'+str(ROOT/'source/runtime'),
                str(ROOT/'tests/thinking_budget_wire.cpp'),
                str(ROOT/'source/runtime/engine/Protocol.cpp'), '-o', str(binary)],
                check=True, capture_output=True)
            frame = self.frame(thinking_budget_tokens=31, thinking_end_token_id=99)
            data = wire.serialize_message(frame)
            result = subprocess.run([str(binary)], input=data, capture_output=True, check=True)
            self.assertEqual(result.stdout, b'31 99 128 2 0')
            image = replace(frame, flags=wire.RequestFlag.VISION_EMBEDDINGS,
                image_spans=(wire.ImageSpan(0, 1, 2, 2, 0, 0),),
                image_pixels=bytes(5120*2), image_embedding_width=5120)
            result = subprocess.run([str(binary)], input=wire.serialize_message(image), capture_output=True, check=True)
            self.assertEqual(result.stdout, b'31 99 128 2 5120')
            invalid = bytearray(data)
            struct.pack_into('<I', invalid, 16+90, 0)
            self.assertNotEqual(subprocess.run([str(binary)], input=invalid, capture_output=True).returncode, 0)
            invalid = bytearray(data)
            struct.pack_into('<H', invalid, 4, 8)
            self.assertNotEqual(subprocess.run([str(binary)], input=invalid, capture_output=True).returncode, 0)


if __name__ == '__main__':
    unittest.main()
