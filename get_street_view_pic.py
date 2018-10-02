import urllib.parse
import urllib.request
import os

myloc = "./street_views"  # replace with your own location
key = "&key=" + ""  # got banned after ~100 requests with no key


def get_street(Add, SaveLoc):
  base = "https://maps.googleapis.com/maps/api/streetview?size=1200x800&location="
  MyUrl = base + urllib.parse.quote_plus(Add) + key  # added url encoding
  fi = Add + ".jpg"
  urllib.request.urlretrieve(MyUrl, os.path.join(SaveLoc,fi))

Tests = ["2643 Virginia Ave NW, Washington, DC 20037",
         "1305 W 7th St, Frederick, MD 21702",
         "2300 Alum Rock Ave, San Jose, CA 95116",
         "634 N Santa Cruz Ave suite 203, Los Gatos, CA 95030",
         "1313 Disneyland Dr, Anaheim, CA 92802"]

for i in Tests:
  get_street(Add=i,SaveLoc=myloc)