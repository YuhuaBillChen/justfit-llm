"""Health probes must finish while their routing caller holds routing_lock."""
import threading
import time
import unittest
from unittest import mock

from ai_service.broker.openai_pool_broker import NodeSpec, PoolState


class ProbeLock(unittest.TestCase):
    def test_idle_llamaswap_fallback_does_not_reacquire_parent_routing_lock(self):
        state = PoolState(nodes=[NodeSpec(name='llamacpp-node',
            base_url='http://127.0.0.1:18034', capabilities=('chat',),
            allowed_models=('*',), host_budget_total=5)])
        state._inflight_reservations['existing'] = ('llamacpp-node', 1, time.time())
        entered, completed = threading.Event(), threading.Event()
        result, errors = [], []

        def inflight(*args):
            entered.set()
            return None

        def probe():
            try:
                result.append(state._cached_health('llamacpp-node', force=True))
            except BaseException as error:
                errors.append(error)
            finally:
                completed.set()

        with (mock.patch.object(state, 'fetch_json_optional', side_effect=[None, {'running': []}]),
              mock.patch.object(state, '_json_request', return_value=(200, b'OK', {})),
              mock.patch.object(state, '_node_is_on_local_host', return_value=True),
              mock.patch.object(state, '_probe_llamaswap_inflight', side_effect=inflight),
              mock.patch.object(state, '_probe_local_amdgpu_system_load', return_value='-')):
            thread = threading.Thread(target=probe)
            with state._routing_lock:
                thread.start()
                started = entered.wait(2)
                finished_while_routing_locked = completed.wait(.5)
            # Release the parent lock before joining, even when the assertion
            # fails, so the regression never leaves a deadlocked test worker.
            thread.join(3)
        self.assertTrue(started)
        self.assertTrue(finished_while_routing_locked,
            'routing waits for this probe, which cannot reacquire the parent routing lock')
        self.assertFalse(thread.is_alive())
        self.assertFalse(errors)
        self.assertEqual(result[0]['active_request_count'], 1)
        self.assertEqual(result[0]['host_budget_used'], 1)


if __name__ == '__main__':
    unittest.main()
