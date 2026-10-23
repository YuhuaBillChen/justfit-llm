"""Bounded startup recovery for an invalid Core AI specialization cache."""
import json
import sys


def load_function(bridge, asset, entrypoint, *, repair_cache=False):
    model = None
    try:
        model = bridge.Model(asset, compute='ane')
        return model, model.function(entrypoint)
    except bridge.BridgeError as error:
        message = str(error).strip()
        cache_error = (message == 'nilError' or message.startswith(
            'failedToPersistResource("Failed to conform existing specialized asset'))
        if not repair_cache or not cache_error:
            raise
    # Leave the exception scope so the failed function traceback can be released.
    del model
    print(json.dumps({'event': 'ane_cache_rebuild', 'attempt': 1}), file=sys.stderr, flush=True)
    bridge.invalidate_model_cache(asset, compute='ane')
    model = bridge.Model(asset, compute='ane')
    return model, model.function(entrypoint)
