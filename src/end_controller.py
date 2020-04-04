import clr

clr.AddReference('ScpDriverInterface')
from ScpDriverInterface import ScpBus

if __name__ == "__main__":
    ScpBus().Unplug(1)