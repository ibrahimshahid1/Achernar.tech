#!/usr/bin/env python3
"""Read-only fresh preview delivery benchmark; no camera settings or commands."""
import argparse
import base64
import hashlib
import json
import os
import socket
import statistics
import struct
import time
import urllib.parse
import urllib.request


def exact(sock, count):
    result = bytearray()
    while len(result) < count:
        chunk = sock.recv(count - len(result))
        if not chunk:
            raise ConnectionError('Preview socket ended')
        result.extend(chunk)
    return bytes(result)


def measure(base, seconds):
    parsed = urllib.parse.urlparse(base)
    if parsed.scheme != 'http' or parsed.hostname not in ('localhost', '127.0.0.1', '::1'):
        raise ValueError('Use the local daemon HTTP URL')
    port = parsed.port or 80
    def get(path):
        with urllib.request.urlopen(base.rstrip('/')+'/api/v1/'+path, timeout=5) as response:
            return json.load(response)
    before=get('status').get('camera')
    if not before or not before.get('streaming'):
        raise RuntimeError('Connect a camera and start live view first')
    with socket.create_connection((parsed.hostname,port), timeout=5) as sock:
        key=base64.b64encode(os.urandom(16)).decode()
        sock.sendall((f'GET /api/v1/camera/stream HTTP/1.1\r\nHost: {parsed.netloc}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n').encode())
        headers=bytearray()
        while not headers.endswith(b'\r\n\r\n'):
            headers.extend(exact(sock,1))
            if len(headers)>16384: raise RuntimeError('Oversized handshake')
        expected=base64.b64encode(hashlib.sha1((key+'258EAFA5-E914-47DA-95CA-C5AB0DC85B11').encode()).digest()).decode()
        if not headers.startswith(b'HTTP/1.1 101') or expected.lower().encode() not in headers.lower():
            raise RuntimeError('Preview WebSocket handshake failed')
        received=[]; source=[]; byte_counts=[]; delivery_age=[]; clears=duplicates=0
        started=time.monotonic()
        while time.monotonic()-started < seconds:
            sock.settimeout(max(.1,seconds-(time.monotonic()-started)))
            try:
                prefix=exact(sock,2); opcode=prefix[0]&15; length=prefix[1]&127
                if length==126: length=struct.unpack('!H',exact(sock,2))[0]
                elif length==127: length=struct.unpack('!Q',exact(sock,8))[0]
                if length>32*1024*1024: raise RuntimeError('Oversized preview packet')
                if prefix[1]&128: raise RuntimeError('Server frame was unexpectedly masked')
                data=exact(sock,length)
            except socket.timeout: break
            if opcode==8: break
            if opcode!=2 or len(data)<20: continue
            width,height,timestamp,kind=struct.unpack('<IIQI',data[:20])
            if not width or not height or kind!=1: clears+=1; continue
            if source and timestamp<=source[-1]: duplicates+=1; continue
            received.append(time.monotonic());source.append(timestamp);byte_counts.append(len(data)-20)
            delivery_age.append(time.time()*1000-timestamp)
    after=get('status').get('camera')
    stable=bool(after and after.get('connection_id')==before.get('connection_id') and after.get('latest_frame',{}).get('stream_generation')==before.get('latest_frame',{}).get('stream_generation'))
    intervals=[(b-a)*1000 for a,b in zip(received,received[1:])]
    source_intervals=[b-a for a,b in zip(source,source[1:])]
    report={'camera':before.get('name'),'driver':before.get('driver_id'),'connection_stable':stable,
        'duration_seconds':seconds,'fresh_frames':len(received),'duplicates_rejected':duplicates,'clear_packets':clears,
        'delivery_fps':(len(received)-1)/(received[-1]-received[0]) if len(received)>1 else None,
        'source_timestamp_fps':1000*(len(source)-1)/(source[-1]-source[0]) if len(source)>1 else None,
        'median_delivery_interval_ms':statistics.median(intervals) if intervals else None,
        'median_source_interval_ms':statistics.median(source_intervals) if source_intervals else None,
        'mean_jpeg_bytes':statistics.mean(byte_counts) if byte_counts else None,
        'median_callback_to_delivery_ms':statistics.median(delivery_age) if delivery_age else None,
        'p95_callback_to_delivery_ms':sorted(delivery_age)[int(.95*(len(delivery_age)-1))] if delivery_age else None,
        'source_counter_delta':after['latest_frame']['frame_counter']-before['latest_frame']['frame_counter'] if stable else None,
        'scope':'Host-stamped preview callback to client delivery; excludes camera exposure and PTP acquisition time. No claim about movie/UVC FPS or cloud accuracy'}
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--daemon-url',default='http://127.0.0.1:3030')
    parser.add_argument('--seconds',type=float,default=10)
    parser.add_argument('--output')
    args=parser.parse_args()
    if not 1<=args.seconds<=60: parser.error('Use a duration between 1 and 60 seconds')
    result=measure(args.daemon_url,args.seconds)
    encoded=json.dumps(result,indent=2)+'\n'
    if args.output:
        with open(args.output,'x') as f:f.write(encoded)
    print(encoded,end='')
