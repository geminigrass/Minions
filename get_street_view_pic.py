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

Tests = ["1712 Berryessa Rd, San Jose, CA 95133"]

for i in Tests:
  get_street(Add=i,SaveLoc=myloc)