import os
import requests
from requests.auth import HTTPDigestAuth
import urllib3
import xml.etree.ElementTree as ET
import time
from dotenv import load_dotenv

load_dotenv()
urllib3.disable_warnings()

# Config — loaded from .env (see .env.example)
IP       = os.getenv("CAMERA_IP")
USERNAME = os.getenv("CAMERA_USERNAME")
PASSWORD = os.getenv("CAMERA_PASSWORD")

if not all([IP, USERNAME, PASSWORD]):
    raise RuntimeError("Missing camera credentials. Copy .env.example → .env and fill in values.")

URL     = f"https://{IP}/onvif/device_service"   # ← ALL services on same endpoint
AUTH    = HTTPDigestAuth(USERNAME, PASSWORD)
HEADERS = {"Content-Type": "application/soap+xml"}

def soap_request(body):
    envelope = f"""<?xml version="1.0" encoding="UTF-8"?>
<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope">
  <s:Body>{body}</s:Body>
</s:Envelope>"""
    r = requests.post(URL, data=envelope, headers=HEADERS,
                      auth=AUTH, verify=False, timeout=10)
    return r.text

def get_profile_token():
    body = '<GetProfiles xmlns="http://www.onvif.org/ver10/media/wsdl"/>'
    resp = soap_request(body)
    root = ET.fromstring(resp)
    profiles = root.findall('.//{http://www.onvif.org/ver10/media/wsdl}Profiles')
    if not profiles:
        # try ver20
        profiles = root.findall('.//{http://www.onvif.org/ver20/media/wsdl}Profiles')
    token = profiles[0].attrib.get('token')
    print(f"✅ Profile token: {token}")
    return token

def get_rtsp_url(token):
    body = f"""<GetStreamUri xmlns="http://www.onvif.org/ver10/media/wsdl">
  <StreamSetup>
    <Stream xmlns="http://www.onvif.org/ver10/schema">RTP-Unicast</Stream>
    <Transport xmlns="http://www.onvif.org/ver10/schema">
      <Protocol>RTSP</Protocol>
    </Transport>
  </StreamSetup>
  <ProfileToken>{token}</ProfileToken>
</GetStreamUri>"""
    resp = soap_request(body)
    root = ET.fromstring(resp)
    uri = root.find('.//{http://www.onvif.org/ver10/schema}Uri')
    if uri is not None:
        print(f"📹 RTSP URL: {uri.text}")
        return uri.text
    print("⚠️ Could not get RTSP URL")
    return None

def ptz_move(token, pan=0.0, tilt=0.0, zoom=0.0):
    body = f"""<ContinuousMove xmlns="http://www.onvif.org/ver20/ptz/wsdl">
  <ProfileToken>{token}</ProfileToken>
  <Velocity>
    <PanTilt xmlns="http://www.onvif.org/ver10/schema" x="{pan}" y="{tilt}"/>
    <Zoom xmlns="http://www.onvif.org/ver10/schema" x="{zoom}"/>
  </Velocity>
</ContinuousMove>"""
    resp = soap_request(body)
    if "Fault" in resp:
        print(f"⚠️  Move fault: {resp[resp.find('<SOAP-ENV:Fault>'):resp.find('</SOAP-ENV:Fault>')+20]}")
    else:
        print(f"✅ Moving pan={pan} tilt={tilt} zoom={zoom}")

def ptz_stop(token):
    body = f"""<Stop xmlns="http://www.onvif.org/ver20/ptz/wsdl">
  <ProfileToken>{token}</ProfileToken>
  <PanTilt>true</PanTilt>
  <Zoom>true</Zoom>
</Stop>"""
    resp = soap_request(body)
    if "Fault" in resp:
        print(f"⚠️  Stop fault: {resp[:300]}")
    else:
        print("🛑 Stopped")

def move_and_stop(token, pan=0.0, tilt=0.0, zoom=0.0, duration=3):
    ptz_move(token, pan, tilt, zoom)
    time.sleep(duration)
    ptz_stop(token)

# --- Main ---
token = get_profile_token()
rtsp  = get_rtsp_url(token)

print("\n➡️  Moving LEFT")
move_and_stop(token, pan=-0.5, duration=3)

print("\n⬅️  Moving RIGHT")
move_and_stop(token, pan=0.5, duration=3)

print("\n⬆️  Moving UP")
move_and_stop(token, tilt=0.5, duration=3)

print("\n⬇️  Moving DOWN")
move_and_stop(token, tilt=-0.5, duration=3)

print("\n🔍 Zoom IN")
move_and_stop(token, zoom=0.5, duration=3)

print("\n🔎 Zoom OUT")
move_and_stop(token, zoom=-0.5, duration=3)

print("\n✅ PTZ test complete")
