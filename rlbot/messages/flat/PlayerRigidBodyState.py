# automatically generated by the FlatBuffers compiler, do not modify

# namespace: flat

import flatbuffers

# /// Rigid body state for a player / car in the game. Includes the latest
# /// controller input, which is otherwise difficult to correlate with consequences.
class PlayerRigidBodyState(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsPlayerRigidBodyState(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = PlayerRigidBodyState()
        x.Init(buf, n + offset)
        return x

    # PlayerRigidBodyState
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # PlayerRigidBodyState
    def State(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            from .RigidBodyState import RigidBodyState
            obj = RigidBodyState()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # PlayerRigidBodyState
    def Input(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            from .ControllerState import ControllerState
            obj = ControllerState()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

def PlayerRigidBodyStateStart(builder): builder.StartObject(2)
def PlayerRigidBodyStateAddState(builder, state): builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(state), 0)
def PlayerRigidBodyStateAddInput(builder, input): builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(input), 0)
def PlayerRigidBodyStateEnd(builder): return builder.EndObject()