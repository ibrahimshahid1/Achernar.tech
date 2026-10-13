#!/usr/bin/env bash
# Read-only checks. Never connects, captures, starts streaming or changes camera settings.
set -euo pipefail
DAEMON_URL="${DAEMON_URL:-http://127.0.0.1:3030}"
DJANGO_URL="${DJANGO_URL:-http://127.0.0.1:8000}"
python3 - "$DAEMON_URL" "$DJANGO_URL" <<'PY'
import json, sys, urllib.request

def request(url, payload=None):
    data=None if payload is None else json.dumps(payload).encode()
    req=urllib.request.Request(url, data=data, headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req, timeout=12) as response:
        return json.load(response)

daemon,django=sys.argv[1:]
registry=request(daemon+'/api/v1/commands')
assert {'coc.status','coc.configure','coc.freeze_sample','coc.save_sample'} <= {r['name'] for r in registry}
direct=request(daemon+'/api/v1/coc/status')
relay=request(django+'/api/equipment/command/',{'action':'coc.status','wait_seconds':8})
assert relay.get('ok') is True, relay
relayed=relay['payload']
for result in (direct,relayed):
    assert result['shadow_only'] is True and result['runtime_eligible'] is False
    assert result['contract_version']=='coc-segmentation-v1'
    assert any(m['loaded'] for m in result['models']), result['models']
assert [(m['id'],m['model_sha256']) for m in direct['models']]==[(m['id'],m['model_sha256']) for m in relayed['models']]
print('PASS: command registry, native research models, direct API and Django relay')
print('Models:', ', '.join(m['id']+(' loaded' if m['loaded'] else ' unavailable') for m in direct['models']))
print('Inference:',direct['status'])
camera=request(daemon+'/api/v1/status').get('camera')
print('Camera:', 'not connected' if not camera else json.dumps({k:camera.get(k) for k in ('name','connected','streaming','last_error','latest_frame')}))
print('Physical night-sky accuracy is not established by these checks.')
PY
