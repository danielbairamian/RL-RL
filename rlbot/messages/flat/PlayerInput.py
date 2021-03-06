# automatically generated by the FlatBuffers compiler, do not modify

# namespace: flat

import flatbuffers

class PlayerInput(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsPlayerInput(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = PlayerInput()
        x.Init(buf, n + offset)
        return x

    # PlayerInput
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # PlayerInput
    def PlayerIndex(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # PlayerInput
    def ControllerState(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            from .ControllerState import ControllerState
            obj = ControllerState()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

def PlayerInputStart(builder): builder.StartObject(2)
def PlayerInputAddPlayerIndex(builder, playerIndex): builder.PrependInt32Slot(0, playerIndex, 0)
def PlayerInputAddControllerState(builder, controllerState): builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(controllerState), 0)
def PlayerInputEnd(builder): return builder.EndObject()
