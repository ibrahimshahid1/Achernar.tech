#!/usr/bin/env python3
"""Observe a user-performed unplug; never connects, streams or changes hardware."""
import argparse
import datetime
import json
import time
import urllib.error
import urllib.request


def request(url, post=False):
    req = urllib.request.Request(url, data=b'{}' if post else None,
                                 headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=4) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode()[:500]


def run(seconds):
    started = time.monotonic()
    events = []
    connected_seen = False
    loss_at = None
    last_key = None
    while time.monotonic() - started < seconds:
        direct = request('http://127.0.0.1:3030/api/v1/status')[1]
        relay = request('http://127.0.0.1:8000/api/equipment/daemon-status/')[1]
        coc = request('http://127.0.0.1:3030/api/v1/coc/status')[1]
        camera = direct.get('camera') or {}
        relayed = relay.get('camera') or {}
        latest = coc.get('latest') or {}
        event = {
            'elapsed_seconds': round(time.monotonic() - started, 3),
            'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'direct_connected': camera.get('connected') is True,
            'relay_connected': relayed.get('connected') is True,
            'daemon_connected': relay.get('daemon_connected'),
            'name': camera.get('name'), 'relay_name': relayed.get('name'),
            'connection_id': camera.get('connection_id'),
            'streaming': camera.get('streaming'),
            'has_frame': camera.get('latest_frame') is not None,
            'relay_has_frame': relayed.get('latest_frame') is not None,
            'health': direct.get('device_health', {}).get('camera'),
            'relay_health': relay.get('device_health', {}).get('camera'),
            'coc_status': coc.get('status'),
            'has_coc_sample': bool(latest),
            'coc_sample_id': latest.get('sample_id'),
        }
        events.append(event)
        key = tuple(event[k] for k in ('direct_connected', 'relay_connected',
                                       'coc_status', 'has_frame', 'has_coc_sample'))
        if key != last_key:
            print(json.dumps(event), flush=True)
            last_key = key
        connected_seen |= event['direct_connected'] and event['relay_connected']
        cleared = (connected_seen and not event['direct_connected']
                   and not event['relay_connected'] and event['daemon_connected']
                   and not event['name'] and not event['relay_name']
                   and not event['has_frame'] and not event['relay_has_frame']
                   and not event['has_coc_sample'])
        if cleared:
            if loss_at is None:
                loss_at = time.monotonic()
            if time.monotonic() - loss_at >= 5:
                snapshots = {}
                for label, url in (
                    ('direct', 'http://127.0.0.1:3030/api/v1/camera/snapshot'),
                    ('django', 'http://127.0.0.1:8000/api/equipment/camera/snapshot/'),
                ):
                    code, body = request(url, post=True)
                    snapshots[label] = {'http_status': code, 'response': body}
                rejected = all(value['http_status'] >= 400 for value in snapshots.values())
                return {'result': 'PASS' if rejected else 'FAIL', 'events': events,
                        'post_loss_snapshots': snapshots,
                        'scope': 'API clearing after a user-performed unplug; elapsed time is from monitor start, not physical unplug'}
        else:
            loss_at = None
        time.sleep(.3)
    return {'result': 'UNKNOWN', 'events': events,
            'reason': 'No sustained connected-to-cleared transition within observation window'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=180)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if not 10 <= args.seconds <= 600:
        parser.error('Use a duration between 10 and 600 seconds')
    result = run(args.seconds)
    with open(args.output, 'x') as output:
        json.dump(result, output, indent=2)
        output.write('\n')
    print(json.dumps({'result': result['result'], 'output': args.output}), flush=True)
    raise SystemExit(0 if result['result'] == 'PASS' else 2)
